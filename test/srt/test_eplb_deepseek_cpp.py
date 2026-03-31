import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb import eplb_algorithms
from sglang.srt.eplb.cpp_deepseek import rebalance_experts_cpp
from sglang.srt.eplb.eplb_algorithms import deepseek
from sglang.srt.eplb.expert_location import ExpertLocationMetadata


class TestEplbDeepseekCpp(unittest.TestCase):
    def test_cpp_matches_python_hierarchical(self):
        weight = torch.tensor(
            [
                [10, 8, 7, 5, 4, 3, 2, 1],
                [4, 9, 3, 8, 2, 7, 1, 6],
            ],
            dtype=torch.float32,
        )

        actual = rebalance_experts_cpp(
            weight=weight,
            num_replicas=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
            enable_hierarchical=True,
        )
        expected = deepseek.rebalance_experts(
            weight=weight,
            num_replicas=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
            enable_hierarchical=True,
        )

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_cpp_matches_python_global(self):
        weight = torch.tensor(
            [
                [10, 8, 7, 5, 4, 3, 2, 1],
                [4, 9, 3, 8, 2, 7, 1, 6],
            ],
            dtype=torch.float32,
        )

        actual = rebalance_experts_cpp(
            weight=weight,
            num_replicas=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
            enable_hierarchical=False,
        )
        expected = deepseek.rebalance_experts(
            weight=weight,
            num_replicas=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
            enable_hierarchical=False,
        )

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_async_path_prefers_cpp_deepseek(self):
        logical_count = torch.ones((1, 2, 8), dtype=torch.float32)
        server_args = SimpleNamespace(
            eplb_algorithm="deepseek_hierarchical",
            nnodes=2,
            enable_eplb_async=True,
            device="cpu",
        )
        common = {
            "model_config_for_expert_location": SimpleNamespace(
                num_groups=4,
                num_logical_experts=8,
            ),
            "num_physical_experts": 16,
            "ep_size": 4,
        }
        cpp_outputs = (
            torch.zeros((2, 16), dtype=torch.int64),
            torch.zeros((2, 8, 2), dtype=torch.int64),
            torch.ones((2, 8), dtype=torch.int64),
        )

        with (
            patch.object(ExpertLocationMetadata, "_init_common", return_value=common),
            patch.object(
                ExpertLocationMetadata, "_init_raw", return_value="cpp-selected"
            ),
            patch(
                "sglang.srt.eplb.expert_location.eplb_algorithms.compute_algorithm",
                return_value=eplb_algorithms.EplbAlgorithm.deepseek_hierarchical,
            ),
            patch(
                "sglang.srt.eplb.expert_location.rebalance_experts_cpp",
                return_value=cpp_outputs,
            ) as cpp_mock,
            patch(
                "sglang.srt.eplb.expert_location.eplb_algorithms.rebalance_experts"
            ) as py_mock,
        ):
            result = ExpertLocationMetadata.init_by_eplb(
                server_args, object(), logical_count
            )

        self.assertEqual(result, "cpp-selected")
        cpp_mock.assert_called_once()
        py_mock.assert_not_called()

    def test_async_path_moves_gpu_logical_count_to_pinned_cpu_in_init(self):
        logical_count = torch.ones((1, 2, 8), dtype=torch.float32)
        logical_count_gpu = Mock()
        logical_count_gpu.shape = logical_count.shape
        logical_count_gpu.dtype = logical_count.dtype
        logical_count_gpu.device = torch.device("cuda", 0)
        logical_count_gpu.is_pinned.return_value = False
        logical_count_cpu = torch.ones((1, 2, 8), dtype=torch.float32, pin_memory=True)
        server_args = SimpleNamespace(
            eplb_algorithm="deepseek_hierarchical",
            nnodes=2,
            enable_eplb_async=True,
            device="cpu",
        )
        common = {
            "model_config_for_expert_location": SimpleNamespace(
                num_groups=4,
                num_logical_experts=8,
            ),
            "num_physical_experts": 16,
            "ep_size": 4,
        }
        cpp_outputs = (
            torch.zeros((2, 16), dtype=torch.int64),
            torch.zeros((2, 8, 2), dtype=torch.int64),
            torch.ones((2, 8), dtype=torch.int64),
        )

        with (
            patch.object(ExpertLocationMetadata, "_init_common", return_value=common),
            patch.object(
                ExpertLocationMetadata, "_init_raw", return_value="cpp-selected"
            ),
            patch(
                "sglang.srt.eplb.expert_location.eplb_algorithms.compute_algorithm",
                return_value=eplb_algorithms.EplbAlgorithm.deepseek_hierarchical,
            ),
            patch(
                "sglang.srt.eplb.expert_location._copy_logical_count_to_pinned_cpu",
                return_value=logical_count_cpu,
            ) as copy_mock,
            patch(
                "sglang.srt.eplb.expert_location.rebalance_experts_cpp",
                return_value=cpp_outputs,
            ) as cpp_mock,
        ):
            result = ExpertLocationMetadata.init_by_eplb(
                server_args, object(), logical_count_gpu
            )

        self.assertEqual(result, "cpp-selected")
        copy_mock.assert_called_once_with(logical_count_gpu)
        cpp_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
