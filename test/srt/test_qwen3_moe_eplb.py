from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    _Buffer,
    AsyncRebalanceSnapshot,
    _convert_global_physical_count_to_logical_count,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.utils import LazyValue


def test_qwen3_moe_eplb_uses_realized_layer_mapping():
    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    model._routed_experts_weights_of_layer = LazyValue(lambda: {4: object(), 2: object()})

    manager = EPLBManager.__new__(EPLBManager)
    manager._model_runner = SimpleNamespace(model=model)
    manager._rebalance_layers_per_chunk = 1

    assert manager._compute_update_layer_ids_chunks() == [[2], [4]]


def test_async_post_launch_prepare_is_disabled():
    manager = EPLBManager.__new__(EPLBManager)
    assert manager._should_use_post_launch_async_prepare() is False


def test_post_launch_async_prepare_submits_on_forward_pass_end():
    recorder = SimpleNamespace(
        skip_next_forward_pass=Mock(),
        prepare_async_rebalance_snapshot=Mock(return_value="snapshot"),
    )
    future = Mock()
    future.done.return_value = False
    executor = Mock()
    executor.submit.return_value = future

    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(enable_eplb_async=True)
    manager._rebalance_num_iterations = 4
    manager._use_post_launch_async_prepare = True
    manager._pending_rebalance_snapshot = None
    manager._prepare_future = None
    manager._prepare_future_target_forward_pass_id = None
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._prepared_rebalance_target_forward_pass_id = None
    manager._prepare_executor = executor
    manager._prepare_stream = None
    manager._post_launch_submitted_forward_pass_id = None
    manager._model_runner = SimpleNamespace(forward_pass_id=3)
    manager._materialize_async_rebalance_logical_count_snapshot = Mock(
        return_value=("logical-count", None, 0.2)
    )
    manager._finish_async_rebalance_prepare = Mock()
    manager._maybe_collect_prepared_async_rebalance = Mock()
    with patch(
        "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
        return_value=recorder,
    ):
        manager.on_forward_pass_end()

        recorder.skip_next_forward_pass.assert_not_called()
        recorder.prepare_async_rebalance_snapshot.assert_called_once()
        assert manager._pending_rebalance_snapshot == "snapshot"

        manager._model_runner.forward_pass_id = 4
        manager.on_forward_graph_launched()

    manager._materialize_async_rebalance_logical_count_snapshot.assert_called_once_with(
        "snapshot"
    )
    executor.submit.assert_called_once_with(
        manager._finish_async_rebalance_prepare,
        ("logical-count", None, 0.2),
    )
    assert manager._prepare_future is future
    assert manager._prepare_future_target_forward_pass_id == 4
    assert manager._pending_rebalance_snapshot is None


def test_post_launch_async_prepare_applies_once_future_is_ready():
    recorder = SimpleNamespace(
        skip_next_forward_pass=Mock(),
        prepare_async_rebalance_snapshot=Mock(return_value=None),
    )
    future = Mock()
    future.done.return_value = True
    future.result.return_value = ("prepared-metadata", "apply-event")

    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(enable_eplb_async=True)
    manager._rebalance_num_iterations = 4
    manager._use_post_launch_async_prepare = True
    manager._prepare_future = future
    manager._prepare_future_target_forward_pass_id = 4
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._prepared_rebalance_target_forward_pass_id = None
    manager._pending_rebalance_snapshot = None
    manager._model_runner = SimpleNamespace(forward_pass_id=4)
    manager._compute_update_layer_ids_chunks = Mock(return_value=[[2], [4]])
    manager._apply_prepared_async_rebalance = Mock()

    with patch(
        "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
        return_value=recorder,
    ):
        manager.on_forward_pass_end()

    assert manager._prepared_rebalance_metadata == "prepared-metadata"
    assert manager._prepared_rebalance_apply_event == "apply-event"
    assert manager._prepared_update_layer_ids_chunks == [[2], [4]]
    assert manager._prepared_rebalance_target_forward_pass_id == 5
    manager._apply_prepared_async_rebalance.assert_not_called()

    manager._model_runner.forward_pass_id = 5
    with patch(
        "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
        return_value=recorder,
    ):
        manager.on_forward_pass_end()

    manager._apply_prepared_async_rebalance.assert_called_once()


def test_post_launch_async_prepare_does_not_apply_in_submit_forward_end():
    recorder = SimpleNamespace(
        skip_next_forward_pass=Mock(),
        prepare_async_rebalance_snapshot=Mock(return_value="snapshot"),
    )
    future = Mock()
    future.done.return_value = True
    future.result.return_value = "prepared-metadata"
    executor = Mock()
    executor.submit.return_value = future

    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(enable_eplb_async=True)
    manager._rebalance_num_iterations = 4
    manager._use_post_launch_async_prepare = True
    manager._pending_rebalance_snapshot = "snapshot"
    manager._prepare_future = None
    manager._prepare_future_target_forward_pass_id = None
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._prepared_rebalance_target_forward_pass_id = None
    manager._prepare_executor = executor
    manager._prepare_stream = None
    manager._post_launch_submitted_forward_pass_id = None
    manager._model_runner = SimpleNamespace(forward_pass_id=4)
    manager._materialize_async_rebalance_logical_count_snapshot = Mock(
        return_value=("logical-count", None, 0.2)
    )
    manager._finish_async_rebalance_prepare = Mock()
    manager._compute_update_layer_ids_chunks = Mock(return_value=[[2], [4]])
    manager._apply_prepared_async_rebalance = Mock()

    with patch(
        "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
        return_value=recorder,
    ):
        manager.on_forward_pass_end()

    manager._apply_prepared_async_rebalance.assert_not_called()
    assert manager._prepare_future is None
    assert manager._prepare_future_target_forward_pass_id is None


def test_async_rebalance_applies_on_next_forward_end():
    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(enable_eplb_async=True)
    manager._use_post_launch_async_prepare = False
    manager._rebalance_num_iterations = 4
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._model_runner = SimpleNamespace(forward_pass_id=4)
    manager._prepare_async_rebalance = Mock(
        side_effect=lambda: (
            setattr(manager, "_prepared_rebalance_metadata", "prepared-metadata"),
            setattr(manager, "_prepared_rebalance_apply_event", "apply-event"),
            setattr(manager, "_prepared_update_layer_ids_chunks", [[2], [4]]),
        )
    )
    manager._apply_prepared_async_rebalance = Mock()
    manager._async_main_generator = manager._async_entrypoint()

    manager.on_forward_pass_end()

    manager._prepare_async_rebalance.assert_called_once()
    manager._apply_prepared_async_rebalance.assert_not_called()

    manager._model_runner.forward_pass_id = 5
    manager.on_forward_pass_end()

    manager._apply_prepared_async_rebalance.assert_called_once()


def test_apply_prepared_async_rebalance_waits_apply_event():
    apply_event = Mock()

    manager = EPLBManager.__new__(EPLBManager)
    manager._prepared_rebalance_metadata = "prepared-metadata"
    manager._prepared_rebalance_apply_event = apply_event
    manager._prepared_update_layer_ids_chunks = [[2], [4]]
    manager._prepared_rebalance_target_forward_pass_id = 5
    manager._model_runner = SimpleNamespace(update_expert_location=Mock())

    manager._apply_prepared_async_rebalance()

    apply_event.synchronize.assert_called_once()
    manager._model_runner.update_expert_location.assert_any_call(
        "prepared-metadata",
        update_layer_ids=[2],
    )
    manager._model_runner.update_expert_location.assert_any_call(
        "prepared-metadata",
        update_layer_ids=[4],
    )
    assert manager._prepared_rebalance_metadata is None
    assert manager._prepared_rebalance_apply_event is None


def test_async_logical_count_uses_per_step_mapping():
    global_physical_count = torch.tensor(
        [
            [[3, 5]],
            [[3, 5]],
        ],
        dtype=torch.int32,
    )
    physical_to_logical_map = torch.tensor(
        [
            [[0, 1]],
            [[1, 0]],
        ],
        dtype=torch.int32,
    )

    logical_count = _convert_global_physical_count_to_logical_count(
        global_physical_count=global_physical_count,
        num_layers=1,
        num_logical_experts=2,
        physical_to_logical_map=physical_to_logical_map,
    )

    assert torch.equal(
        logical_count,
        torch.tensor(
            [
                [[3, 5]],
                [[5, 3]],
            ],
            dtype=torch.int32,
        ),
    )


def test_async_mapping_buffer_uses_pinned_cpu_memory():
    buffer = _Buffer.init_new(
        item_shape=(2, 4),
        buffer_size=2,
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,
    )

    buffer.append(torch.arange(8, dtype=torch.int32).view(2, 4))
    all_values = buffer.get_all()

    assert all_values.device.type == "cpu"
    assert all_values.is_pinned()
def test_materialize_async_rebalance_uses_frozen_snapshot_buffers():
    snapshot = AsyncRebalanceSnapshot(
        num_logical_experts=2,
        average_utilization_rate_over_window=0.3,
    )

    manager = EPLBManager.__new__(EPLBManager)
    manager._prepare_stream = None
    global_physical_count = torch.ones((2, 1, 2), dtype=torch.int32)
    physical_to_logical_map = torch.zeros((2, 1, 2), dtype=torch.int32)

    with patch(
        "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
        return_value=SimpleNamespace(
            detach_async_rebalance_global_physical_count=Mock(
                return_value=global_physical_count
            ),
            detach_async_rebalance_physical_to_logical_map=Mock(
                return_value=physical_to_logical_map
            ),
        ),
    ):
        result = manager._materialize_async_rebalance_logical_count_snapshot(snapshot)

    assert result[0] is global_physical_count
    assert result[1] is physical_to_logical_map
    assert result[2] == 2
    assert result[3] is None
    assert result[4] == 0.3


def test_init_async_prepare_expert_location_metadata_runs_on_prepare_stream():
    stream_context = Mock()
    stream_context.__enter__ = Mock(return_value=None)
    stream_context.__exit__ = Mock(return_value=None)
    prepare_stream = Mock()
    apply_event = Mock()
    logical_count_cpu = torch.ones((1, 1, 2), dtype=torch.int32)

    manager = EPLBManager.__new__(EPLBManager)
    manager._prepare_stream = prepare_stream
    manager._server_args = object()
    manager._model_runner = SimpleNamespace(model_config=object())

    with (
        patch("sglang.srt.eplb.eplb_manager.torch.cuda.stream", return_value=stream_context),
        patch("sglang.srt.eplb.eplb_manager.torch.cuda.Event", return_value=apply_event),
        patch(
            "sglang.srt.eplb.eplb_manager.ExpertLocationMetadata.init_by_eplb",
            return_value="metadata",
        ) as init_by_eplb,
    ):
        result = manager._init_async_prepare_expert_location_metadata(logical_count_cpu)

    init_by_eplb.assert_called_once_with(
        manager._server_args,
        manager._model_runner.model_config,
        logical_count_cpu,
    )
    apply_event.record.assert_called_once_with(prepare_stream)
    apply_event.synchronize.assert_not_called()
    assert result == ("metadata", apply_event)


def test_materialize_async_rebalance_kicks_off_cuda_d2h_on_prepare_stream():
    stream_context = Mock()
    stream_context.__enter__ = Mock(return_value=None)
    stream_context.__exit__ = Mock(return_value=None)
    prepare_stream = Mock()
    ready_event = Mock()
    copy_done_event = Mock()
    snapshot = AsyncRebalanceSnapshot(
        num_logical_experts=2,
        average_utilization_rate_over_window=0.3,
        ready_event=ready_event,
    )

    manager = EPLBManager.__new__(EPLBManager)
    manager._prepare_stream = prepare_stream
    cuda_tensor = Mock()
    cuda_tensor.is_cuda = True
    physical_to_logical_map = torch.zeros((2, 1, 2), dtype=torch.int32)
    global_physical_count_cpu = Mock()

    with (
        patch(
            "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
            return_value=SimpleNamespace(
                detach_async_rebalance_global_physical_count=Mock(return_value=cuda_tensor),
                detach_async_rebalance_physical_to_logical_map=Mock(
                    return_value=physical_to_logical_map
                ),
            ),
        ),
        patch("sglang.srt.eplb.eplb_manager.torch.empty", return_value=global_physical_count_cpu),
        patch("sglang.srt.eplb.eplb_manager.torch.cuda.stream", return_value=stream_context),
        patch("sglang.srt.eplb.eplb_manager.torch.cuda.Event", return_value=copy_done_event),
    ):
        result = manager._materialize_async_rebalance_logical_count_snapshot(snapshot)

    prepare_stream.wait_event.assert_called_once_with(ready_event)
    global_physical_count_cpu.copy_.assert_called_once_with(cuda_tensor, non_blocking=True)
    copy_done_event.record.assert_called_once_with(prepare_stream)
    assert result[0] is global_physical_count_cpu
    assert result[1] is physical_to_logical_map
    assert result[2] == 2
    assert result[3] is copy_done_event
    assert result[4] == 0.3


def test_finish_async_rebalance_prepare_moves_logical_count_d2h_to_prepare_stream():
    stream_context = Mock()
    stream_context.__enter__ = Mock(return_value=None)
    stream_context.__exit__ = Mock(return_value=None)
    prepare_stream = Mock()
    logical_count_cuda = Mock()
    logical_count_cuda.device.type = "cuda"
    logical_count_cuda.shape = (2, 1, 2)
    logical_count_cuda.dtype = torch.int32
    logical_count_cpu = Mock()
    apply_event = Mock()

    manager = EPLBManager.__new__(EPLBManager)
    manager._prepare_stream = prepare_stream
    manager._server_args = object()
    manager._model_runner = SimpleNamespace(model_config=object())
    manager._check_rebalance_needed = Mock(return_value=True)

    with (
        patch(
            "sglang.srt.eplb.eplb_manager._convert_global_physical_count_to_logical_count",
            return_value=logical_count_cuda,
        ),
        patch("sglang.srt.eplb.eplb_manager.torch.empty", return_value=logical_count_cpu),
        patch("sglang.srt.eplb.eplb_manager.torch.cuda.stream", return_value=stream_context),
        patch("sglang.srt.eplb.eplb_manager.torch.cuda.Event", return_value=apply_event),
        patch(
            "sglang.srt.eplb.eplb_manager.ExpertLocationMetadata.init_by_eplb",
            return_value="metadata",
        ) as init_by_eplb,
    ):
        result = manager._finish_async_rebalance_prepare(
            (
                torch.ones((2, 1, 2), dtype=torch.int32),
                torch.zeros((2, 1, 2), dtype=torch.int32),
                2,
                None,
                0.3,
            )
    )

    logical_count_cpu.copy_.assert_called_once_with(logical_count_cuda, non_blocking=True)
    apply_event.record.assert_called_once_with(prepare_stream)
    apply_event.synchronize.assert_not_called()
    init_by_eplb.assert_called_once_with(
        manager._server_args,
        manager._model_runner.model_config,
        logical_count_cpu,
    )
    assert result == ("metadata", apply_event)
