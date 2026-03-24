import unittest
from typing import Tuple

import torch

from sglang.srt.eplb import eplb_algorithms
from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm, deepseek


def _reference_balanced_packing_cpu(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def _reference_rebalance_experts_hierarchical_cpu(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = _reference_balanced_packing_cpu(
        tokens_per_group, num_nodes
    )
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = deepseek.replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = _reference_balanced_packing_cpu(
        tokens_per_phy, num_gpus // num_nodes
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def _reference_rebalance_experts_cpu(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
):
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if enable_hierarchical:
        phy2log, phyrank, logcnt = _reference_rebalance_experts_hierarchical_cpu(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, phyrank, logcnt = _reference_rebalance_experts_hierarchical_cpu(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


class TestEplbAlgorithms(unittest.TestCase):
    def test_deepseek_algorithm_is_not_cpu_only(self):
        self.assertFalse(eplb_algorithms.algorithm_runs_on_cpu(EplbAlgorithm.deepseek))

    def test_balanced_packing_defaults_to_gpu_implementation(self):
        weight = torch.tensor(
            [[10, 8, 7, 5, 4, 3, 2, 1]],
            dtype=torch.float32,
        )

        actual = deepseek.balanced_packing(weight, num_packs=4)
        expected = deepseek._balanced_packing_gpu_impl(weight, num_packs=4)

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_balanced_packing_gpu_falls_back_to_impl_without_triton_gpu(self):
        weight = torch.tensor(
            [[10, 8, 7, 5, 4, 3, 2, 1]],
            dtype=torch.float32,
        )

        actual = deepseek.balanced_packing_gpu(weight, num_packs=4)
        expected = deepseek._balanced_packing_gpu_impl(weight, num_packs=4)

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_deepseek_keeps_original_cpu_balanced_packing(self):
        weight = torch.tensor(
            [
                [10, 8, 7, 5, 4, 3, 2, 1],
                [4, 9, 3, 8, 2, 7, 1, 6],
            ],
            dtype=torch.float32,
        )

        actual = deepseek.balanced_packing_cpu(weight, num_packs=4)
        expected = _reference_balanced_packing_cpu(weight, num_packs=4)

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_deepseek_global_path_matches_original_cpu_algorithm(self):
        weight = torch.tensor(
            [
                [10, 8, 7, 5, 4, 3, 2, 1],
                [4, 9, 3, 8, 2, 7, 1, 6],
            ],
            dtype=torch.float32,
        )

        actual = deepseek.rebalance_experts(
            weight=weight,
            num_replicas=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
            enable_hierarchical=False,
        )
        expected = _reference_rebalance_experts_cpu(
            weight=weight,
            num_replicas=16,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
            enable_hierarchical=False,
        )

        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2].cpu(), expected[2], rtol=0, atol=0)

        actual_log2phy = actual[1].cpu()
        expected_log2phy = expected[1]
        expected_width = expected_log2phy.shape[-1]
        torch.testing.assert_close(
            actual_log2phy[:, :, :expected_width], expected_log2phy, rtol=0, atol=0
        )
        self.assertTrue(torch.all(actual_log2phy[:, :, expected_width:] == -1))


if __name__ == "__main__":
    unittest.main()
