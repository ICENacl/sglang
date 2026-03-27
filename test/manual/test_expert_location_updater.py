import os
import time
import traceback
import unittest
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from sglang.srt.eplb import expert_location_updater
from sglang.srt.eplb.cpp_async_runtime import create_eplb_async_runtime
from sglang.srt.eplb.eplb_async_host_mirror import (
    EPLBAsyncHostMirrorManager,
    _build_node_availability_from_metadata,
    _build_cross_node_transfer_plan,
    _compute_local_owner_indices,
    _group_cross_node_transfer_plan,
    _compute_node_owner_physical_ids,
)
from sglang.test.test_utils import CustomTestCase, find_available_port
from sglang.utils import is_in_ci


@dataclass
class _TestInfo:
    nnodes: int
    num_logical_experts: int
    num_physical_experts: int
    num_repeat: int = 5000


class TestExpertLocationUpdater(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_cpu(self):
        self._test_common(device="cpu")
        self._test_core(
            num_gpus=32,
            device="cpu",
            infos=[
                _TestInfo(
                    nnodes=4,
                    num_logical_experts=256,
                    num_physical_experts=288,
                    num_repeat=10000,
                )
            ],
        )

    def test_cpu_slow(self):
        if is_in_ci():
            return
        self._test_core(
            num_gpus=144,
            device="cpu",
            infos=[
                _TestInfo(
                    nnodes=18,
                    num_logical_experts=256,
                    num_physical_experts=288,
                    num_repeat=10000,
                )
            ],
        )

    def test_gpu(self):
        if is_in_ci():
            return
        self._test_common(device="cuda")

    def test_async_runtime_signal_cuda_manual(self):
        if is_in_ci() or not torch.cuda.is_available():
            return

        runtime = create_eplb_async_runtime(torch.cuda.current_device())
        try:
            runtime.register_layer(0)

            runtime.prepare_capture_step(1)
            enabled = runtime.wait_gpu_stage(0)
            torch.cuda.current_stream().synchronize()
            self.assertEqual(int(enabled.cpu().item()), 0)

            runtime.start_iter(2, True)
            time.sleep(0.01)
            enabled = runtime.wait_gpu_stage(0)
            runtime.set_cpu_stage(0)
            torch.cuda.current_stream().synchronize()
            self.assertEqual(int(enabled.cpu().item()), 1)

            runtime.wait_for_idle()
        finally:
            runtime.shutdown()

    def test_async_host_mirror_single_rank_cpu(self):
        old_physical_to_logical_map = [0, 1, 2, 3]
        new_physical_to_logical_map = [3, 1, 0, 2]
        routed_experts_weights = [
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor(
                [[0, 0], [1, 10], [2, 20], [3, 30]],
                dtype=torch.int64,
            ),
        ]
        host_mirror = {
            logical_expert_id: [
                torch.tensor(logical_expert_id, dtype=torch.int64),
                torch.tensor(
                    [logical_expert_id, logical_expert_id * 10],
                    dtype=torch.int64,
                ),
            ]
            for logical_expert_id in range(4)
        }

        expert_location_updater.update_expert_weights_single_layer_from_logical_host_mirror_direct(
            routed_experts_weights=routed_experts_weights,
            old_physical_to_logical_map=old_physical_to_logical_map,
            new_physical_to_logical_map=new_physical_to_logical_map,
            num_local_physical_experts=4,
            num_gpu_per_node=1,
            rank=0,
            host_expert_tensors_getter=lambda logical_expert_id: host_mirror[
                logical_expert_id
            ],
            copy_stream=None,
            layer_main_done_event=None,
            layer_weights_ready_event=None,
        )

        self.assertTrue(
            torch.equal(
                routed_experts_weights[0],
                torch.tensor([3, 1, 0, 2], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                routed_experts_weights[1],
                torch.tensor([[3, 30], [1, 10], [0, 0], [2, 20]], dtype=torch.int64),
            )
        )

    def test_async_host_mirror_cycle_fallback_cpu(self):
        old_physical_to_logical_map = [0, 1]
        new_physical_to_logical_map = [1, 0]
        routed_experts_weights = [
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([[100, 1000], [200, 2000]], dtype=torch.int64),
        ]
        host_mirror = {
            0: [
                torch.tensor(0, dtype=torch.int64),
                torch.tensor([100, 1000], dtype=torch.int64),
            ],
            1: [
                torch.tensor(1, dtype=torch.int64),
                torch.tensor([200, 2000], dtype=torch.int64),
            ],
        }

        debug_logs = expert_location_updater.update_expert_weights_single_layer_from_logical_host_mirror_direct(
            routed_experts_weights=routed_experts_weights,
            old_physical_to_logical_map=old_physical_to_logical_map,
            new_physical_to_logical_map=new_physical_to_logical_map,
            num_local_physical_experts=2,
            num_gpu_per_node=1,
            rank=0,
            host_expert_tensors_getter=lambda logical_expert_id: host_mirror[
                logical_expert_id
            ],
            copy_stream=None,
            layer_main_done_event=None,
            layer_weights_ready_event=None,
            debug=True,
        )

        self.assertTrue(
            torch.equal(routed_experts_weights[0], torch.tensor([1, 0], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                routed_experts_weights[1],
                torch.tensor([[200, 2000], [100, 1000]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            any("case=host-mirror-fallback" in log for log in debug_logs),
            debug_logs,
        )

    def test_async_host_mirror_free_rider_cpu(self):
        old_physical_to_logical_map = [0, 4, 5]
        new_physical_to_logical_map = [1, 1, 5]
        routed_experts_weights = [
            torch.tensor([0, 4, 5], dtype=torch.int64),
            torch.tensor([[0, 0], [4, 40], [5, 50]], dtype=torch.int64),
        ]
        host_mirror = {
            1: [
                torch.tensor(1, dtype=torch.int64),
                torch.tensor([1, 10], dtype=torch.int64),
            ],
            5: [
                torch.tensor(5, dtype=torch.int64),
                torch.tensor([5, 50], dtype=torch.int64),
            ],
        }

        debug_logs = expert_location_updater.update_expert_weights_single_layer_from_logical_host_mirror_direct(
            routed_experts_weights=routed_experts_weights,
            old_physical_to_logical_map=old_physical_to_logical_map,
            new_physical_to_logical_map=new_physical_to_logical_map,
            num_local_physical_experts=3,
            num_gpu_per_node=1,
            rank=0,
            host_expert_tensors_getter=lambda logical_expert_id: host_mirror[
                logical_expert_id
            ],
            copy_stream=None,
            layer_main_done_event=None,
            layer_weights_ready_event=None,
            debug=True,
        )

        self.assertTrue(
            torch.equal(routed_experts_weights[0], torch.tensor([1, 1, 5], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                routed_experts_weights[1],
                torch.tensor([[1, 10], [1, 10], [5, 50]], dtype=torch.int64),
            )
        )
        self.assertTrue(any("case=free-rider" in log for log in debug_logs), debug_logs)

    def test_async_host_mirror_node_owner_selection(self):
        owners = _compute_node_owner_physical_ids(
            torch.tensor([3, 1, 3, 0, 2, 1], dtype=torch.int64),
            node_physical_start=0,
            node_physical_end=6,
            num_logical_experts=4,
        )
        self.assertEqual(owners, [3, 1, 4, 0])

    def test_async_host_mirror_cross_node_transfer_plan(self):
        availability = torch.tensor(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.uint8,
        )
        self.assertEqual(
            _build_cross_node_transfer_plan(availability),
            [
                (0, 0, 1),
                (0, 0, 2),
                (1, 1, 0),
                (1, 1, 2),
                (2, 0, 1),
                (2, 0, 2),
                (3, 2, 0),
                (3, 2, 1),
            ],
        )

    def test_async_host_mirror_node_availability_from_metadata(self):
        availability = _build_node_availability_from_metadata(
            torch.tensor([0, 1, 2, 1, 3, 0, 3, 4], dtype=torch.int64),
            num_nodes=2,
            local_world_size=2,
            num_local_physical_experts=2,
            num_logical_experts=5,
        )
        self.assertTrue(
            torch.equal(
                availability,
                torch.tensor(
                    [
                        [1, 1, 1, 0, 0],
                        [1, 0, 0, 1, 1],
                    ],
                    dtype=torch.uint8,
                ),
            )
        )

    def test_async_host_mirror_grouped_transfer_plan(self):
        availability = torch.tensor(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.uint8,
        )
        self.assertEqual(
            _group_cross_node_transfer_plan(
                _build_cross_node_transfer_plan(availability)
            ),
            [
                (0, 1, (0, 2)),
                (0, 2, (0, 2)),
                (1, 0, (1,)),
                (1, 2, (1,)),
                (2, 0, (3,)),
                (2, 1, (3,)),
            ],
        )

    def test_async_host_mirror_local_owner_indices(self):
        local_expert_ids, logical_expert_ids = _compute_local_owner_indices(
            torch.tensor([0, 1, 2, 1, 3, 0, 3, 4], dtype=torch.int64),
            rank=1,
            num_local_physical_experts=2,
            node_physical_start=4,
            node_physical_end=8,
            num_logical_experts=5,
        )
        self.assertTrue(
            torch.equal(local_expert_ids, torch.tensor([0, 1], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(logical_expert_ids, torch.tensor([3, 4], dtype=torch.int64))
        )

    def test_async_host_mirror_dummy_h2d_returns_fake_tensors(self):
        manager = EPLBAsyncHostMirrorManager.__new__(EPLBAsyncHostMirrorManager)
        manager._dummy_h2d = True
        manager._dummy_layer_tensors = {}
        manager._layer_num_tensors = {}
        manager._rank = 0
        manager._registered_atexit = True
        manager._maybe_register_atexit = lambda: None

        routed_experts_weights_of_layer = {
            3: [
                torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
                torch.tensor([[5], [6]], dtype=torch.float32),
            ]
        }

        manager._build_dummy_from_loaded_model(routed_experts_weights_of_layer)
        fake_tensors = manager.get_expert_tensors(3, 1)

        self.assertEqual(len(fake_tensors), 2)
        self.assertTrue(torch.equal(fake_tensors[0], torch.zeros((2,), dtype=torch.float32)))
        self.assertTrue(torch.equal(fake_tensors[1], torch.zeros((1,), dtype=torch.float32)))

    def _test_common(self, device):
        infos = []

        for nnodes in [1, 2, 4]:
            for num_logical_experts in [2, 5, 20, 256]:
                for num_physical_experts in [8, 16, 256, 288]:
                    if num_logical_experts > num_physical_experts:
                        continue
                    infos.append(
                        _TestInfo(
                            nnodes=nnodes,
                            num_logical_experts=num_logical_experts,
                            num_physical_experts=num_physical_experts,
                        )
                    )

        self._test_core(num_gpus=8, device=device, infos=infos)

    def _test_core(
        self,
        num_gpus: int,
        device: str,
        infos: List[_TestInfo],
    ):
        master_port = find_available_port(23456)

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for rank in range(num_gpus):
            p = Process(
                target=_run_subprocess,
                kwargs=dict(
                    rank=rank,
                    num_gpus=num_gpus,
                    output_writer=output_writer,
                    master_port=master_port,
                    device=device,
                    infos=infos,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(num_gpus):
            self.assertTrue(
                output_reader.recv(), f"Subprocess has error, please see logs above."
            )

        for p in processes:
            p.join()


def _run_subprocess(
    rank: int,
    num_gpus: int,
    master_port: int,
    device: str,
    infos: List[_TestInfo],
    output_writer,
):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        torch.random.manual_seed(42)
        torch.distributed.init_process_group(
            rank=rank,
            world_size=num_gpus,
            backend={"cpu": "gloo", "cuda": None}[device],
        )
        if device == "cuda":
            torch.cuda.set_device(f"cuda:{rank}")
        if device == "xpu":
            torch.xpu.set_device(f"xpu:{rank}")

        for info in infos:
            _execute_test(info, rank=rank, num_gpus=num_gpus, device=device)

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


def _execute_test(info: _TestInfo, rank: int, num_gpus: int, device: str):
    if rank == 0:
        print(f"Test: {num_gpus=} {info=}", flush=True)

    assert info.num_physical_experts % num_gpus == 0
    num_local_physical_experts = info.num_physical_experts // num_gpus
    assert num_gpus % info.nnodes == 0
    num_gpu_per_node = num_gpus // info.nnodes

    def _create_routed_experts_weights(physical_to_logical_map):
        local_logical_expert_ids = physical_to_logical_map[
            rank * num_local_physical_experts : (rank + 1) * num_local_physical_experts
        ].cpu()
        return [
            local_logical_expert_ids.to(device).clone(),
            torch.tensor(
                [
                    [local_logical_expert_id * 10, local_logical_expert_id * 100]
                    for local_logical_expert_id in local_logical_expert_ids.tolist()
                ],
                device=device,
            ),
        ]

    def _create_physical_to_logical_map():
        if rank == 0:
            ans = torch.concat(
                [
                    torch.arange(0, info.num_logical_experts),
                    torch.randint(
                        0,
                        info.num_logical_experts,
                        (info.num_physical_experts - info.num_logical_experts,),
                    ),
                ]
            )
            ans = ans[torch.randperm(ans.shape[0])]
        else:
            ans = torch.empty((info.num_physical_experts,), dtype=torch.int64)

        assert ans.dtype == torch.int64 and ans.shape == (info.num_physical_experts,)
        ans = ans.to(device)
        torch.distributed.broadcast(ans, src=0)

        return ans.cpu()

    physical_to_logical_map = _create_physical_to_logical_map()
    routed_experts_weights = _create_routed_experts_weights(physical_to_logical_map)

    for i in range(info.num_repeat):
        if rank == 0 and ((i % 500 == 0) or (i == info.num_repeat - 1)):
            print(f"Step {i}/{info.num_repeat}", flush=True)

        new_physical_to_logical_map = _create_physical_to_logical_map()
        expect_new_weights = _create_routed_experts_weights(new_physical_to_logical_map)

        output_logs = expert_location_updater.update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights,
            temp_buffers=expert_location_updater.create_temp_buffers(
                routed_experts_weights
            ),
            old_physical_to_logical_map=physical_to_logical_map.tolist(),
            new_physical_to_logical_map=new_physical_to_logical_map.tolist(),
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            debug=True,
        )

        local_has_error = not all(
            torch.all(x == y)
            for x, y in zip(routed_experts_weights, expect_new_weights, strict=True)
        )
        global_has_error = torch.tensor(local_has_error, device=device)
        torch.distributed.all_reduce(
            global_has_error, op=torch.distributed.ReduceOp.MAX
        )

        if global_has_error.cpu().item():
            output_logs_str = "\n".join(output_logs)
            local_message = (
                f"===================== rank {rank} ============================\n"
                f"{num_gpus=} {info=}\n"
                f"{routed_experts_weights[0].tolist()=}\n"
                f"{expect_new_weights[0].tolist()=}\n"
                f"{physical_to_logical_map.tolist()=}\n"
                f"{new_physical_to_logical_map.tolist()=}\n"
                f"===logs===\n"
                f"{output_logs_str}\n"
                f"==============================================================\n"
            )

            global_messages = ([None] * num_gpus) if rank == 0 else None
            torch.distributed.gather_object(local_message, global_messages, dst=0)

            if rank == 0:
                print("\n\n".join(global_messages), flush=True)
            raise AssertionError(f"Error happens, see logs above")

        physical_to_logical_map = new_physical_to_logical_map


if __name__ == "__main__":
    unittest.main()
