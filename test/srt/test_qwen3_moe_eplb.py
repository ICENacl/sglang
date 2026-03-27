from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
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


def test_post_launch_async_prepare_submits_after_graph_launch():
    recorder = SimpleNamespace(
        skip_next_forward_pass=Mock(),
        prepare_async_rebalance_snapshot=Mock(return_value="snapshot"),
        detach_async_rebalance_global_physical_count=Mock(return_value="buffer"),
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

        recorder.skip_next_forward_pass.assert_called_once()
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
    future.result.return_value = "prepared-metadata"

    manager = EPLBManager.__new__(EPLBManager)
    manager._server_args = SimpleNamespace(enable_eplb_async=True)
    manager._rebalance_num_iterations = 4
    manager._use_post_launch_async_prepare = True
    manager._prepare_future = future
    manager._prepare_future_target_forward_pass_id = 4
    manager._prepared_rebalance_metadata = None
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
    assert manager._prepared_update_layer_ids_chunks == [[2], [4]]
    assert manager._prepared_rebalance_target_forward_pass_id == 4
    manager._apply_prepared_async_rebalance.assert_called_once()


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
