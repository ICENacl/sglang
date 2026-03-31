from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    _Buffer,
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


def test_async_rebalance_applies_on_next_forward_end():
    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(enable_eplb_async=True)
    manager._rebalance_num_iterations = 4
    manager._pending_logical_count = None
    manager._pending_logical_count_ready_event = None
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._model_runner = SimpleNamespace(forward_pass_id=4)
    manager._start_async_rebalance_logical_count_fetch = Mock(
        side_effect=lambda: (
            setattr(manager, "_pending_logical_count", "logical-count-gpu"),
            setattr(manager, "_pending_logical_count_ready_event", Mock(query=Mock(return_value=True))),
        )
    )
    manager._prepare_async_rebalance = Mock(
        side_effect=lambda: (
            setattr(manager, "_pending_logical_count", None),
            setattr(manager, "_pending_logical_count_ready_event", None),
            setattr(manager, "_prepared_rebalance_metadata", "prepared-metadata"),
            setattr(manager, "_prepared_rebalance_apply_event", "apply-event"),
            setattr(manager, "_prepared_update_layer_ids_chunks", [[2], [4]]),
        )
    )
    manager._apply_prepared_async_rebalance = Mock()
    manager._async_main_generator = manager._async_entrypoint()

    manager.on_forward_pass_end()

    manager._start_async_rebalance_logical_count_fetch.assert_called_once()
    manager._prepare_async_rebalance.assert_not_called()
    manager._apply_prepared_async_rebalance.assert_not_called()

    manager._model_runner.forward_pass_id = 5
    manager.on_forward_pass_end()

    manager._prepare_async_rebalance.assert_called_once()
    manager._apply_prepared_async_rebalance.assert_not_called()

    manager._model_runner.forward_pass_id = 6
    manager.on_forward_pass_end()

    manager._apply_prepared_async_rebalance.assert_called_once()


def test_apply_prepared_async_rebalance_waits_apply_event():
    apply_event = Mock()

    manager = EPLBManager.__new__(EPLBManager)
    manager._prepared_rebalance_metadata = "prepared-metadata"
    manager._prepared_rebalance_apply_event = apply_event
    manager._prepared_update_layer_ids_chunks = [[2], [4]]
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


def test_async_mapping_buffer_uses_gpu_memory():
    device = "cuda" if torch.cuda.is_available() else "meta"
    buffer = _Buffer.init_new(
        item_shape=(2, 4),
        buffer_size=2,
        dtype=torch.int32,
        device=device,
    )
    assert buffer.get_all().device.type == torch.device(device).type


def test_init_async_prepare_expert_location_metadata_runs_on_prepare_stream():
    stream_context = Mock()
    stream_context.__enter__ = Mock(return_value=None)
    stream_context.__exit__ = Mock(return_value=None)
    prepare_stream = Mock()
    apply_event = Mock()
    logical_count_ready_event = Mock()
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
        result = manager._init_async_prepare_expert_location_metadata(
            logical_count_cpu,
            logical_count_ready_event=logical_count_ready_event,
        )

    init_by_eplb.assert_called_once_with(
        manager._server_args,
        manager._model_runner.model_config,
        logical_count_cpu,
    )
    prepare_stream.wait_event.assert_called_once_with(logical_count_ready_event)
    apply_event.record.assert_called_once_with(prepare_stream)
    apply_event.synchronize.assert_not_called()
    assert result == ("metadata", apply_event)


def test_prepare_async_rebalance_waits_for_pending_logical_count_ready():
    ready_event = Mock()
    ready_event.query.return_value = False

    manager = EPLBManager.__new__(EPLBManager)
    manager._pending_logical_count = torch.ones((1, 1, 2), dtype=torch.int32)
    manager._pending_logical_count_ready_event = ready_event
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._server_args = object()
    manager._model_runner = SimpleNamespace(model_config=object(), model=SimpleNamespace())

    manager._prepare_async_rebalance()

    ready_event.query.assert_called_once()
    assert manager._pending_logical_count is not None
    assert manager._prepared_rebalance_metadata is None


def test_start_async_rebalance_logical_count_fetch_keeps_gpu_tensor():
    logical_count = Mock()
    logical_count.device.type = "cuda"
    ready_event = Mock()

    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(
        enable_eplb_async=True,
        eplb_min_rebalancing_utilization_threshold=1.0,
    )
    manager._prepared_rebalance_metadata = None
    manager._prepared_rebalance_apply_event = None
    manager._prepared_update_layer_ids_chunks = None
    manager._pending_logical_count = None
    manager._pending_logical_count_ready_event = None

    with (
        patch(
            "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder"
        ) as recorder_getter,
        patch.object(
            manager, "_record_logical_count_ready_event", return_value=ready_event
        ) as record_ready_event,
    ):
        recorder = recorder_getter.return_value
        recorder.dump_record.return_value = {
            "logical_count": logical_count,
            "average_utilization_rate_over_window": None,
        }
        manager._start_async_rebalance_logical_count_fetch()

    recorder.materialize_async_snapshot.assert_called_once()
    recorder.dump_record.assert_called_once_with(output_mode="object")
    record_ready_event.assert_called_once_with(logical_count)
    assert manager._pending_logical_count is logical_count
    assert manager._pending_logical_count_ready_event is ready_event
