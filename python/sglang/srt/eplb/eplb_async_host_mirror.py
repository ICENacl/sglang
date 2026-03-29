import atexit
import ctypes
import contextlib
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from sglang.srt.distributed import get_world_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

logger = logging.getLogger(__name__)


def _sanitize_name(raw: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", raw)


def _get_host_mirror_model_name(model_config) -> str:
    architectures = getattr(model_config.hf_config, "architectures", None)
    if architectures and len(architectures) > 0:
        return architectures[0]
    return os.path.basename(model_config.model_path.rstrip("/")) or model_config.model_path


def _element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _format_nbytes(num_bytes: int) -> str:
    return f"{num_bytes} B ({num_bytes / (1024 * 1024 * 1024):.2f} GB)"


def _compute_node_owner_physical_ids(
    physical_to_logical_map: torch.Tensor,
    *,
    node_physical_start: int,
    node_physical_end: int,
    num_logical_experts: int,
) -> list[int]:
    owners = [-1] * num_logical_experts
    for global_physical_expert_id in range(node_physical_start, node_physical_end):
        logical_expert_id = int(physical_to_logical_map[global_physical_expert_id].item())
        if owners[logical_expert_id] == -1:
            owners[logical_expert_id] = global_physical_expert_id
    return owners


def _build_cross_node_transfer_plan(
    availability: torch.Tensor,
) -> list[tuple[int, int, int]]:
    plan = []
    num_nodes, num_logical_experts = availability.shape
    for logical_expert_id in range(num_logical_experts):
        src_nodes = torch.nonzero(availability[:, logical_expert_id], as_tuple=False).flatten()
        if src_nodes.numel() == 0:
            continue
        src_node_rank = int(src_nodes[0].item())
        for dst_node_rank in range(num_nodes):
            if int(availability[dst_node_rank, logical_expert_id].item()) == 0:
                plan.append((logical_expert_id, src_node_rank, dst_node_rank))
    return plan


def _build_node_availability_from_metadata(
    physical_to_logical_map: torch.Tensor,
    *,
    num_nodes: int,
    local_world_size: int,
    num_local_physical_experts: int,
    num_logical_experts: int,
) -> torch.Tensor:
    availability = torch.zeros((num_nodes, num_logical_experts), dtype=torch.uint8)
    experts_per_node = local_world_size * num_local_physical_experts
    for node_rank in range(num_nodes):
        node_physical_start = node_rank * experts_per_node
        node_physical_end = node_physical_start + experts_per_node
        owners = _compute_node_owner_physical_ids(
            physical_to_logical_map,
            node_physical_start=node_physical_start,
            node_physical_end=node_physical_end,
            num_logical_experts=num_logical_experts,
        )
        for logical_expert_id, global_physical_expert_id in enumerate(owners):
            if global_physical_expert_id != -1:
                availability[node_rank, logical_expert_id] = 1
    return availability


def _group_cross_node_transfer_plan(
    transfer_plan: list[tuple[int, int, int]],
) -> list[tuple[int, int, tuple[int, ...]]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for logical_expert_id, src_node_rank, dst_node_rank in transfer_plan:
        grouped.setdefault((src_node_rank, dst_node_rank), []).append(logical_expert_id)
    return [
        (src_node_rank, dst_node_rank, tuple(logical_expert_ids))
        for (src_node_rank, dst_node_rank), logical_expert_ids in sorted(grouped.items())
    ]


def _compute_local_owner_indices(
    physical_to_logical_map: torch.Tensor,
    *,
    rank: int,
    num_local_physical_experts: int,
    node_physical_start: int,
    node_physical_end: int,
    num_logical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_owner_physical_ids = _compute_node_owner_physical_ids(
        physical_to_logical_map,
        node_physical_start=node_physical_start,
        node_physical_end=node_physical_end,
        num_logical_experts=num_logical_experts,
    )
    local_physical_start = rank * num_local_physical_experts
    local_expert_ids = []
    logical_expert_ids = []
    for local_expert_id in range(num_local_physical_experts):
        global_physical_expert_id = local_physical_start + local_expert_id
        logical_expert_id = int(physical_to_logical_map[global_physical_expert_id].item())
        if node_owner_physical_ids[logical_expert_id] == global_physical_expert_id:
            local_expert_ids.append(local_expert_id)
            logical_expert_ids.append(logical_expert_id)
    return (
        torch.tensor(local_expert_ids, dtype=torch.int64),
        torch.tensor(logical_expert_ids, dtype=torch.int64),
    )


@dataclass
class _ShmRecord:
    shm: shared_memory.SharedMemory
    tensor: torch.Tensor
    name: str
    is_owner: bool
    unlink_on_close: bool


class EPLBAsyncHostMirrorManager:
    def __init__(self, server_args, model_config):
        config = ModelConfigForExpertLocation.from_model_config(model_config)
        assert config is not None, "EPLB async requires a MoE model."

        self._server_args = server_args
        self._world_group = get_world_group()
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._local_rank = self._world_group.local_rank
        self._local_world_size = self._world_group.local_size or max(
            1, self._world_size // max(1, server_args.nnodes)
        )
        self._node_rank = server_args.node_rank
        self._num_nodes = server_args.nnodes
        self._is_owner = self._local_rank == 0
        self._num_logical_experts = config.num_logical_experts
        self._records: dict[tuple[int, int], _ShmRecord] = {}
        self._valid_records: dict[int, _ShmRecord] = {}
        self._layer_tensors: dict[int, list[torch.Tensor]] = {}
        self._dummy_layer_tensors: dict[int, list[torch.Tensor]] = {}
        self._layer_num_tensors: dict[int, int] = {}
        self._gpu_staging_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._cpu_transfer_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._leader_device_group = None
        self._leader_cpu_group = None
        self._leader_global_ranks: list[int] = []
        self._closed = False
        self._registered_atexit = False
        self._lock = threading.Lock()
        self._reuse_existing_shm = envs.SGLANG_EPLB_ASYNC_HOST_MIRROR_REUSE_SHM.get()
        self._dummy_h2d = envs.SGLANG_EPLB_ASYNC_DUMMY_H2D.get()

        model_name = _sanitize_name(_get_host_mirror_model_name(model_config))
        master_port = _sanitize_name(os.environ.get("MASTER_PORT", "0"))
        self._base_name = (
            f"sglang_eplb_async_{model_name}_p{master_port}_n{server_args.node_rank}"
        )
        self._maybe_create_leader_groups()

    def build_from_loaded_model(self, routed_experts_weights_of_layer) -> None:
        if self._dummy_h2d:
            self._build_dummy_from_loaded_model(routed_experts_weights_of_layer)
            return

        metadata = get_global_expert_location_metadata()
        assert metadata is not None, "EPLB async host mirror requires expert metadata."

        build_start = time.time()
        num_layers = len(routed_experts_weights_of_layer)
        num_tensor_records = sum(
            len(tensors) for tensors in routed_experts_weights_of_layer.values()
        )
        num_local_fill_steps = num_layers * metadata.num_local_physical_experts
        layer_transfer_plan = self._build_grouped_cross_node_transfer_plan(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            metadata=metadata,
        )
        (
            local_remote_transfer_steps,
            remote_transfer_experts,
            remote_transfer_bytes,
        ) = self._summarize_transfer_plan(
            layer_transfer_plan,
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
        )
        pbar = tqdm(
            total=(
                num_layers
                + num_tensor_records
                + num_local_fill_steps
                + local_remote_transfer_steps
                + num_tensor_records
            ),
            desc="Building EPLB async host mirror",
            disable=self._rank != 0,
            dynamic_ncols=True,
        )
        reused_existing_data = False
        phase_times = {
            "create_records": 0.0,
            "populate_local": 0.0,
            "remote_transfer": 0.0,
            "attach": 0.0,
        }

        try:
            phase_start = time.time()
            self._create_records(routed_experts_weights_of_layer, pbar=pbar)
            phase_times["create_records"] = time.time() - phase_start
            reused_existing_data = self._can_reuse_existing_data()
            if reused_existing_data:
                pbar.update(num_local_fill_steps + local_remote_transfer_steps)
                remote_transfer_experts = 0
                remote_transfer_bytes = 0
            else:
                phase_start = time.time()
                self._populate_local_node_shards(
                    routed_experts_weights_of_layer=routed_experts_weights_of_layer,
                    metadata=metadata,
                    pbar=pbar,
                )
                phase_times["populate_local"] = time.time() - phase_start
                self._barrier_all_ranks()
                phase_start = time.time()
                self._fill_missing_from_remote_nodes(
                    layer_transfer_plan=layer_transfer_plan,
                    pbar=pbar,
                )
                phase_times["remote_transfer"] = time.time() - phase_start
                self._barrier_all_ranks()
            self._validate_completeness()

            phase_start = time.time()
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                attached = []
                for tensor_index, _ in enumerate(tensors):
                    record = self._records.get((layer_id, tensor_index))
                    if record is None:
                        raise RuntimeError(
                            f"EPLB async host mirror is incomplete for layer={layer_id} tensor_index={tensor_index}."
                        )
                    attached.append(record.tensor)
                    pbar.update(1)
                self._layer_tensors[layer_id] = attached
            phase_times["attach"] = time.time() - phase_start
        finally:
            pbar.close()

        self._maybe_register_atexit()
        if self._rank == 0:
            shm_nbytes = self._get_total_shm_nbytes()
            logger.info(
                "EPLB async host mirror build complete: reused_existing_data=%s "
                "layers=%s tensor_records=%s remote_transfer_experts=%s "
                "remote_transfer_bytes=%s shm=%s "
                "create_records=%.2fs populate_local=%.2fs remote_transfer=%.2fs "
                "attach=%.2fs elapsed=%.2fs",
                reused_existing_data,
                num_layers,
                num_tensor_records,
                remote_transfer_experts,
                _format_nbytes(remote_transfer_bytes),
                _format_nbytes(shm_nbytes),
                phase_times["create_records"],
                phase_times["populate_local"],
                phase_times["remote_transfer"],
                phase_times["attach"],
                time.time() - build_start,
            )

    def get_expert_tensors(self, layer_id: int, logical_expert_id: int):
        if self._dummy_h2d:
            if layer_id not in self._dummy_layer_tensors:
                raise KeyError(
                    f"Layer {layer_id} does not exist in async dummy host mirror."
                )
            return self._dummy_layer_tensors[layer_id]
        if layer_id not in self._layer_tensors:
            raise KeyError(f"Layer {layer_id} does not exist in async host mirror.")
        return [tensor[logical_expert_id] for tensor in self._layer_tensors[layer_id]]

    def close(self):
        if self._closed:
            return
        self._closed = True

        for record in self._records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
            if record.unlink_on_close:
                with contextlib.suppress(FileNotFoundError):
                    record.shm.unlink()
        for record in self._valid_records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
            if record.unlink_on_close:
                with contextlib.suppress(FileNotFoundError):
                    record.shm.unlink()
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                if self._leader_device_group is not None:
                    dist.destroy_process_group(self._leader_device_group)
            with contextlib.suppress(Exception):
                if self._leader_cpu_group is not None:
                    dist.destroy_process_group(self._leader_cpu_group)

        self._records.clear()
        self._valid_records.clear()
        self._layer_tensors.clear()
        self._dummy_layer_tensors.clear()
        self._layer_num_tensors.clear()
        self._gpu_staging_buffers.clear()
        self._cpu_transfer_buffers.clear()
        self._leader_device_group = None
        self._leader_cpu_group = None

    def _build_dummy_from_loaded_model(self, routed_experts_weights_of_layer) -> None:
        self._dummy_layer_tensors.clear()
        self._layer_num_tensors.clear()

        for layer_id, tensors in routed_experts_weights_of_layer.items():
            self._layer_num_tensors[layer_id] = len(tensors)
            fake_tensors = []
            for tensor in tensors:
                fake = torch.empty_like(tensor[0], device="cpu", pin_memory=True)
                fake.zero_()
                fake_tensors.append(fake)
            self._dummy_layer_tensors[layer_id] = fake_tensors

        self._maybe_register_atexit()
        if self._rank == 0:
            logger.info(
                "EPLB async dummy H2D enabled: skipped host mirror build, layers=%s tensor_records=%s",
                len(routed_experts_weights_of_layer),
                sum(len(tensors) for tensors in routed_experts_weights_of_layer.values()),
            )

    def _maybe_register_atexit(self):
        if self._registered_atexit:
            return
        self._registered_atexit = True
        atexit.register(self.close)

    def _create_records(self, routed_experts_weights_of_layer, pbar=None) -> None:
        for layer_id, tensors in routed_experts_weights_of_layer.items():
            self._layer_num_tensors[layer_id] = len(tensors)
        if self._is_owner:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._get_or_create_valid_tensor(layer_id)
                if pbar is not None:
                    pbar.update(1)
                for tensor_index, tensor in enumerate(tensors):
                    self._get_or_create_tensor(
                        layer_id=layer_id,
                        tensor_index=tensor_index,
                        sample_tensor=tensor,
                    )
                    if pbar is not None:
                        pbar.update(1)
        self._barrier_all_ranks()
        if not self._is_owner:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._get_or_create_valid_tensor(layer_id)
                for tensor_index, tensor in enumerate(tensors):
                    self._get_or_create_tensor(
                        layer_id=layer_id,
                        tensor_index=tensor_index,
                        sample_tensor=tensor,
                    )

    def _populate_local_node_shards(
        self, *, routed_experts_weights_of_layer, metadata, pbar=None
    ) -> None:
        num_local_physical_experts = metadata.num_local_physical_experts
        node_physical_start = self._node_rank * self._local_world_size * num_local_physical_experts
        node_physical_end = node_physical_start + self._local_world_size * num_local_physical_experts

        for layer_id, tensors in routed_experts_weights_of_layer.items():
            local_expert_ids_cpu, logical_expert_ids_cpu = _compute_local_owner_indices(
                metadata.physical_to_logical_map_cpu[layer_id],
                rank=self._rank,
                num_local_physical_experts=num_local_physical_experts,
                node_physical_start=node_physical_start,
                node_physical_end=node_physical_end,
                num_logical_experts=self._num_logical_experts,
            )
            if local_expert_ids_cpu.numel() == 0:
                continue
            valid_tensor = self._valid_records[layer_id].tensor
            local_expert_ids = local_expert_ids_cpu.to(device=tensors[0].device)
            for tensor_index, tensor in enumerate(tensors):
                record = self._records[(layer_id, tensor_index)]
                packed = tensor.index_select(0, local_expert_ids).cpu()
                record.tensor.index_copy_(0, logical_expert_ids_cpu, packed)
            valid_tensor.index_fill_(0, logical_expert_ids_cpu, 1)
            if pbar is not None:
                pbar.update(local_expert_ids_cpu.numel())

    def _fill_missing_from_remote_nodes(self, *, layer_transfer_plan, pbar=None) -> None:
        if self._leader_cpu_group is None:
            return
        if not self._is_owner:
            return

        for layer_id in sorted(layer_transfer_plan):
            transfer_groups = layer_transfer_plan[layer_id]
            if len(transfer_groups) == 0:
                continue

            for src_node_rank, dst_node_rank, logical_expert_ids in transfer_groups:
                if self._node_rank not in (src_node_rank, dst_node_rank):
                    continue
                self._transfer_logical_expert_group(
                    layer_id=layer_id,
                    src_node_rank=src_node_rank,
                    dst_node_rank=dst_node_rank,
                    logical_expert_ids=logical_expert_ids,
                )
                if pbar is not None:
                    pbar.update(self._layer_num_tensors[layer_id])

            dist.barrier(group=self._leader_cpu_group)

    def _transfer_logical_expert_group(
        self,
        *,
        layer_id: int,
        src_node_rank: int,
        dst_node_rank: int,
        logical_expert_ids: tuple[int, ...],
    ) -> None:
        assert self._leader_global_ranks
        src_global_rank = self._leader_global_ranks[src_node_rank]
        dst_global_rank = self._leader_global_ranks[dst_node_rank]
        logical_expert_ids_cpu = torch.tensor(logical_expert_ids, dtype=torch.int64)
        num_logical_experts = len(logical_expert_ids)

        if self._node_rank == dst_node_rank:
            valid_tensor = self._valid_records[layer_id].tensor
        else:
            valid_tensor = None

        for tensor_index in range(self._layer_num_tensors[layer_id]):
            record = self._records[(layer_id, tensor_index)]
            gpu_staging, cpu_transfer = self._get_or_create_transfer_buffers(
                layer_id=layer_id,
                tensor_index=tensor_index,
                num_logical_experts=num_logical_experts,
            )
            if self._node_rank == src_node_rank:
                gpu_staging.copy_(
                    record.tensor.index_select(0, logical_expert_ids_cpu),
                    non_blocking=False,
                )
                self._send_tensor(gpu_staging, dst_global_rank)
            elif self._node_rank == dst_node_rank:
                self._recv_tensor(gpu_staging, src_global_rank)
                cpu_transfer.copy_(gpu_staging, non_blocking=False)
                record.tensor.index_copy_(0, logical_expert_ids_cpu, cpu_transfer)

        if valid_tensor is not None:
            valid_tensor.index_fill_(0, logical_expert_ids_cpu, 1)

    def _send_tensor(self, tensor: torch.Tensor, dst_global_rank: int) -> None:
        if tensor.is_cuda:
            dist.send(tensor, dst=dst_global_rank, group=self._leader_device_group)
        else:
            dist.send(tensor, dst=dst_global_rank, group=self._leader_cpu_group)

    def _recv_tensor(self, tensor: torch.Tensor, src_global_rank: int) -> None:
        if tensor.is_cuda:
            dist.recv(tensor, src=src_global_rank, group=self._leader_device_group)
        else:
            dist.recv(tensor, src=src_global_rank, group=self._leader_cpu_group)

    def _validate_completeness(self) -> None:
        for layer_id, valid_record in self._valid_records.items():
            missing = torch.nonzero(valid_record.tensor == 0, as_tuple=False).flatten().tolist()
            if missing:
                raise RuntimeError(
                    "EPLB async host mirror is incomplete after initialization: "
                    f"node_rank={self._node_rank} layer_id={layer_id} missing_logical_expert_ids={missing}"
                )

    def _barrier_all_ranks(self) -> None:
        if dist.is_initialized():
            self._world_group.barrier()

    def _maybe_create_leader_groups(self) -> None:
        if (not dist.is_initialized()) or self._num_nodes <= 1:
            return

        self._leader_global_ranks = [
            node_rank * self._local_world_size for node_rank in range(self._num_nodes)
        ]
        backend = dist.get_backend(self._world_group.device_group)
        self._leader_device_group = dist.new_group(
            self._leader_global_ranks,
            backend=backend,
        )
        self._leader_cpu_group = dist.new_group(
            self._leader_global_ranks,
            backend="gloo",
        )

    def _get_or_create_tensor(
        self,
        *,
        layer_id: int,
        tensor_index: int,
        sample_tensor: torch.Tensor,
    ) -> torch.Tensor:
        key = (layer_id, tensor_index)
        if key in self._records:
            return self._records[key].tensor

        shm_name = self._tensor_shm_name(layer_id, tensor_index)
        num_bytes = self._num_bytes_for_tensor(sample_tensor)
        shm, is_owner, unlink_on_close, created_new = self._open_or_create_shm(
            shm_name, num_bytes
        )
        tensor = _create_buffer_tensor(
            shm=shm,
            shape=self._mirror_shape(sample_tensor),
            dtype=sample_tensor.dtype,
        )
        if created_new:
            tensor.zero_()
        _cuda_host_register_tensor(tensor)
        self._records[key] = _ShmRecord(
            shm=shm,
            tensor=tensor,
            name=shm_name,
            is_owner=is_owner,
            unlink_on_close=unlink_on_close,
        )
        self._gpu_staging_buffers[key] = torch.empty(
            (1, *tensor.shape[1:]),
            dtype=sample_tensor.dtype,
            device=sample_tensor.device,
        )
        self._cpu_transfer_buffers[key] = torch.empty(
            (1, *tensor.shape[1:]),
            dtype=sample_tensor.dtype,
            pin_memory=True,
        )
        return tensor

    def _get_or_create_valid_tensor(self, layer_id: int) -> torch.Tensor:
        if layer_id in self._valid_records:
            return self._valid_records[layer_id].tensor

        shm_name = self._valid_shm_name(layer_id)
        num_bytes = self._num_logical_experts * _element_size(torch.uint8)
        shm, is_owner, unlink_on_close, created_new = self._open_or_create_shm(
            shm_name, num_bytes
        )
        tensor = _create_buffer_tensor(
            shm=shm,
            shape=(self._num_logical_experts,),
            dtype=torch.uint8,
        )
        if created_new:
            tensor.zero_()
        self._valid_records[layer_id] = _ShmRecord(
            shm=shm,
            tensor=tensor,
            name=shm_name,
            is_owner=is_owner,
            unlink_on_close=unlink_on_close,
        )
        return tensor

    def _mirror_shape(self, tensor: torch.Tensor):
        return (self._num_logical_experts, *tensor.shape[1:])

    def _num_bytes_for_tensor(self, tensor: torch.Tensor) -> int:
        return math.prod(self._mirror_shape(tensor)) * _element_size(tensor.dtype)

    def _tensor_shm_name(self, layer_id: int, tensor_index: int) -> str:
        return f"{self._base_name}_l{layer_id}_t{tensor_index}"

    def _valid_shm_name(self, layer_id: int) -> str:
        return f"{self._base_name}_l{layer_id}_valid"

    def _open_or_create_shm(self, shm_name: str, num_bytes: int):
        if self._is_owner:
            return self._open_or_create_owner_shm(shm_name, num_bytes)

        for _ in range(200):
            try:
                shm = _open_shared_memory(name=shm_name, track=False)
                self._validate_shm_size(shm, shm_name, num_bytes)
                return shm, False, False, False
            except FileNotFoundError:
                time.sleep(0.05)

        raise RuntimeError(f"Timed out waiting for host mirror shared memory {shm_name}.")

    def _open_or_create_owner_shm(self, shm_name: str, num_bytes: int):
        try:
            shm = _open_shared_memory(
                name=shm_name,
                create=True,
                size=num_bytes,
                track=not self._reuse_existing_shm,
            )
            return shm, True, not self._reuse_existing_shm, True
        except FileExistsError:
            if self._reuse_existing_shm:
                logger.debug(
                    "Reusing existing EPLB async shared memory name: %s",
                    shm_name,
                )
                shm = _open_shared_memory(name=shm_name, track=False)
                self._validate_shm_size(shm, shm_name, num_bytes)
                return shm, True, False, False
            logger.warning(
                "Found existing EPLB async shared memory name: %s, unlinking...",
                shm_name,
            )
            stale = shared_memory.SharedMemory(name=shm_name)
            stale.close()
            stale.unlink()
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)
            return shm, True, True, True

    def _validate_shm_size(self, shm, shm_name: str, num_bytes: int):
        actual_size = getattr(shm, "size", None)
        if actual_size is not None and actual_size != num_bytes:
            shm.close()
            raise RuntimeError(
                f"EPLB async shared memory size mismatch for {shm_name}: "
                f"expected={num_bytes} actual={actual_size}"
            )

    def _can_reuse_existing_data(self) -> bool:
        if not self._reuse_existing_shm:
            return False
        if len(self._valid_records) == 0:
            return False
        for valid_record in self._valid_records.values():
            if not torch.all(valid_record.tensor == 1):
                return False
        return True

    def _get_total_shm_nbytes(self) -> int:
        total = 0
        for record in self._records.values():
            total += getattr(record.shm, "size", 0)
        for record in self._valid_records.values():
            total += getattr(record.shm, "size", 0)
        return total

    def _build_grouped_cross_node_transfer_plan(
        self, *, routed_experts_weights_of_layer, metadata
    ) -> dict[int, list[tuple[int, int, tuple[int, ...]]]]:
        plan = {}
        num_local_physical_experts = metadata.num_local_physical_experts
        for layer_id in sorted(routed_experts_weights_of_layer):
            availability = _build_node_availability_from_metadata(
                metadata.physical_to_logical_map_cpu[layer_id],
                num_nodes=self._num_nodes,
                local_world_size=self._local_world_size,
                num_local_physical_experts=num_local_physical_experts,
                num_logical_experts=self._num_logical_experts,
            )
            plan[layer_id] = _group_cross_node_transfer_plan(
                _build_cross_node_transfer_plan(availability)
            )
        return plan

    def _summarize_transfer_plan(
        self, layer_transfer_plan, *, routed_experts_weights_of_layer
    ) -> tuple[int, int, int]:
        local_steps = 0
        total_experts = 0
        total_bytes = 0
        for layer_id, transfer_groups in layer_transfer_plan.items():
            num_tensors = len(routed_experts_weights_of_layer[layer_id])
            for src_node_rank, dst_node_rank, logical_expert_ids in transfer_groups:
                num_group_experts = len(logical_expert_ids)
                total_experts += num_group_experts
                if self._node_rank in (src_node_rank, dst_node_rank):
                    local_steps += num_tensors
                for tensor in routed_experts_weights_of_layer[layer_id]:
                    total_bytes += (
                        num_group_experts * tensor[0].numel() * tensor.element_size()
                    )
        return local_steps, total_experts, total_bytes

    def _get_or_create_transfer_buffers(
        self, *, layer_id: int, tensor_index: int, num_logical_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_id, tensor_index)
        gpu_staging = self._gpu_staging_buffers[key]
        cpu_transfer = self._cpu_transfer_buffers[key]
        if gpu_staging.shape[0] < num_logical_experts:
            expanded_shape = (num_logical_experts, *gpu_staging.shape[1:])
            self._gpu_staging_buffers[key] = torch.empty(
                expanded_shape,
                dtype=gpu_staging.dtype,
                device=gpu_staging.device,
            )
            self._cpu_transfer_buffers[key] = torch.empty(
                expanded_shape,
                dtype=cpu_transfer.dtype,
                pin_memory=True,
            )
            gpu_staging = self._gpu_staging_buffers[key]
            cpu_transfer = self._cpu_transfer_buffers[key]
        return gpu_staging[:num_logical_experts], cpu_transfer[:num_logical_experts]


def _create_buffer_tensor(
    *,
    shm: shared_memory.SharedMemory,
    shape,
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = math.prod(shape)
    raw = torch.frombuffer(shm.buf, dtype=torch.uint8)
    return raw.view(dtype)[:numel].view(*shape)


def _open_shared_memory(*, name: str, create: bool = False, size: int = 0, track: bool = True):
    if track:
        return shared_memory.SharedMemory(name=name, create=create, size=size)
    with patch.object(resource_tracker, "register", lambda *args, **kwargs: None):
        return shared_memory.SharedMemory(name=name, create=create, size=size)


def _cuda_host_register_tensor(tensor: torch.Tensor):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "EPLB async requires CUDA, but torch.cuda.is_available() is False."
        )

    libcudart = ctypes.CDLL("libcudart.so")
    cuda_host_register_portable = 0x01
    num_bytes = tensor.numel() * tensor.element_size()
    result = libcudart.cudaHostRegister(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(num_bytes),
        ctypes.c_uint(cuda_host_register_portable),
    )
    if result != 0:
        raise RuntimeError(
            f"cudaHostRegister failed with CUDA error code {result} "
            f"for tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )


_INSTANCE = None


def get_global_eplb_async_host_mirror_manager():
    return _INSTANCE


def set_global_eplb_async_host_mirror_manager(value):
    global _INSTANCE
    _INSTANCE = value
