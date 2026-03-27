import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, List

import torch.cuda

from sglang.srt.eplb import eplb_algorithms
from sglang.srt.eplb.expert_distribution import (
    AsyncRebalanceSnapshot,
    _convert_global_physical_count_to_logical_count,
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._use_post_launch_async_prepare = self._should_use_post_launch_async_prepare()
        self._prepared_rebalance_metadata = None
        self._prepared_update_layer_ids_chunks = None
        self._prepared_rebalance_target_forward_pass_id = None
        self._pending_rebalance_snapshot = None
        self._prepare_future: Future | None = None
        self._prepare_future_target_forward_pass_id = None
        self._post_launch_submitted_forward_pass_id = None
        self._prepare_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb-prepare")
            if self._use_post_launch_async_prepare
            else None
        )
        self._prepare_stream = (
            torch.cuda.Stream()
            if self._use_post_launch_async_prepare and torch.cuda.is_available()
            else None
        )
        self._main_generator = self._entrypoint()

    def on_forward_pass_start(self):
        if not self._server_args.enable_eplb_async:
            return

        if self._use_post_launch_async_prepare:
            self._post_launch_submitted_forward_pass_id = None
            return

        if self._model_runner.forward_pass_id % self._rebalance_num_iterations == 0:
            self._prepare_async_rebalance()

    def on_forward_graph_launched(self):
        if not self._server_args.enable_eplb_async:
            return
        if not self._use_post_launch_async_prepare:
            return

        forward_pass_id = self._model_runner.forward_pass_id
        if self._post_launch_submitted_forward_pass_id == forward_pass_id:
            return
        self._post_launch_submitted_forward_pass_id = forward_pass_id

        if forward_pass_id % self._rebalance_num_iterations != 0:
            return
        if self._pending_rebalance_snapshot is None:
            return
        if self._prepare_future is not None:
            return

        logger.info("[EPLBManager] async rebalance post-launch prepare submit")
        logical_count_snapshot = self._materialize_async_rebalance_logical_count_snapshot(
            self._pending_rebalance_snapshot
        )
        target_forward_pass_id = forward_pass_id
        self._pending_rebalance_snapshot = None
        assert self._prepare_executor is not None
        self._prepare_future = self._prepare_executor.submit(
            self._finish_async_rebalance_prepare,
            logical_count_snapshot,
        )
        self._prepare_future_target_forward_pass_id = target_forward_pass_id

    def on_forward_pass_end(self):
        if self._server_args.enable_eplb_async:
            if self._use_post_launch_async_prepare:
                self._handle_post_launch_async_rebalance_on_forward_end()
                return
            if (
                self._model_runner.forward_pass_id % self._rebalance_num_iterations
                == self._rebalance_num_iterations - 1
            ):
                get_global_expert_distribution_recorder().skip_next_forward_pass()
            if self._model_runner.forward_pass_id % self._rebalance_num_iterations == 0:
                self._apply_prepared_async_rebalance()
            return
        next(self._main_generator)

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        mode = "async" if self._server_args.enable_eplb_async else "sync"
        logger.info(f"[EPLBManager] rebalance start mode={mode}")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _prepare_async_rebalance(self):
        logger.info("[EPLBManager] async rebalance prepare start")
        get_global_expert_distribution_recorder().materialize_async_snapshot()
        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            self._prepared_rebalance_metadata = None
            self._prepared_update_layer_ids_chunks = None
            return

        self._prepared_rebalance_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )
        self._prepared_update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        logger.info("[EPLBManager] async rebalance prepare end")

    def _handle_post_launch_async_rebalance_on_forward_end(self):
        forward_pass_id = self._model_runner.forward_pass_id
        if (
            forward_pass_id % self._rebalance_num_iterations
            == self._rebalance_num_iterations - 1
        ):
            get_global_expert_distribution_recorder().skip_next_forward_pass()
            if (
                self._pending_rebalance_snapshot is None
                and self._prepare_future is None
                and self._prepared_rebalance_metadata is None
            ):
                snapshot = (
                    get_global_expert_distribution_recorder().prepare_async_rebalance_snapshot()
                )
                if snapshot is not None and self._check_rebalance_needed(
                    snapshot.average_utilization_rate_over_window
                ):
                    self._pending_rebalance_snapshot = snapshot
            else:
                logger.info(
                    "[EPLBManager] Skip creating new post-launch snapshot because previous async rebalance is still pending"
                )

        self._maybe_collect_prepared_async_rebalance()
        if (
            self._prepared_rebalance_metadata is not None
            and self._prepared_rebalance_target_forward_pass_id is not None
            and forward_pass_id >= self._prepared_rebalance_target_forward_pass_id
        ):
            self._apply_prepared_async_rebalance()

    def _maybe_collect_prepared_async_rebalance(self):
        if self._prepare_future is None or not self._prepare_future.done():
            return

        result = self._prepare_future.result()
        self._prepare_future = None
        target_forward_pass_id = self._prepare_future_target_forward_pass_id
        self._prepare_future_target_forward_pass_id = None
        if result is None:
            return

        self._prepared_rebalance_metadata = result
        self._prepared_update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        self._prepared_rebalance_target_forward_pass_id = target_forward_pass_id

    def _finish_async_rebalance_prepare(self, logical_count_snapshot):
        (
            global_physical_count,
            physical_to_logical_map,
            num_logical_experts,
            ready_event,
            average_utilization_rate_over_window,
        ) = (
            logical_count_snapshot
        )
        if ready_event is not None:
            ready_event.synchronize()

        logical_count = _convert_global_physical_count_to_logical_count(
            global_physical_count=global_physical_count,
            num_layers=global_physical_count.shape[1],
            num_logical_experts=num_logical_experts,
            physical_to_logical_map=physical_to_logical_map,
        )

        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return None

        return ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

    def _materialize_async_rebalance_logical_count_snapshot(
        self,
        snapshot: AsyncRebalanceSnapshot,
    ):
        global_physical_count = (
            get_global_expert_distribution_recorder().detach_async_rebalance_global_physical_count()
        )
        physical_to_logical_map = (
            get_global_expert_distribution_recorder().detach_async_rebalance_physical_to_logical_map()
        )
        if global_physical_count is None:
            raise RuntimeError("Missing async rebalance global physical count buffer.")
        if physical_to_logical_map is None:
            raise RuntimeError("Missing async rebalance physical_to_logical_map buffer.")

        if global_physical_count.is_cuda:
            assert self._prepare_stream is not None
            global_physical_count_cpu = torch.empty(
                global_physical_count.shape,
                dtype=global_physical_count.dtype,
                device="cpu",
                pin_memory=True,
            )
            with torch.cuda.stream(self._prepare_stream):
                if snapshot.ready_event is not None:
                    self._prepare_stream.wait_event(snapshot.ready_event)
                global_physical_count_cpu.copy_(global_physical_count, non_blocking=True)
                ready_event = torch.cuda.Event()
                ready_event.record(self._prepare_stream)
            return (
                global_physical_count_cpu,
                physical_to_logical_map,
                snapshot.num_logical_experts,
                ready_event,
                snapshot.average_utilization_rate_over_window,
            )

        return (
            global_physical_count,
            physical_to_logical_map,
            snapshot.num_logical_experts,
            None,
            snapshot.average_utilization_rate_over_window,
        )

    def _apply_prepared_async_rebalance(self):
        if self._prepared_rebalance_metadata is None:
            return
        logger.info("[EPLBManager] async rebalance apply start")
        for update_layer_ids in self._prepared_update_layer_ids_chunks:
            self._model_runner.update_expert_location(
                self._prepared_rebalance_metadata,
                update_layer_ids=update_layer_ids,
            )
        self._prepared_rebalance_metadata = None
        self._prepared_update_layer_ids_chunks = None
        self._prepared_rebalance_target_forward_pass_id = None
        logger.info("[EPLBManager] async rebalance apply end")

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > self._server_args.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {self._server_args.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _should_use_post_launch_async_prepare(self) -> bool:
        if not self._server_args.enable_eplb_async:
            return False
        if self._server_args.expert_distribution_recorder_mode not in [
            "stat",
            "stat_approx",
        ]:
            return False

        common = ExpertLocationMetadata._init_common(
            self._server_args, self._model_runner.model_config
        )
        if common is None:
            return False

        algorithm = eplb_algorithms.compute_algorithm(
            raw_algorithm=self._server_args.eplb_algorithm,
            num_groups=common["model_config_for_expert_location"].num_groups,
            num_nodes=self._server_args.nnodes,
        )
        return algorithm in [
            eplb_algorithms.EplbAlgorithm.deepseek,
            eplb_algorithms.EplbAlgorithm.deepseek_hierarchical,
        ]


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
