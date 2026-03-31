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
