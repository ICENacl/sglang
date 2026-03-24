# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def balanced_packing_cpu(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
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
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


if triton is not None:

    @triton.jit
    def _balanced_packing_step_triton(
        weight_ptr,
        indices_ptr,
        pack_weights_ptr,
        pack_items_ptr,
        pack_index_ptr,
        rank_in_pack_ptr,
        step,
        num_packs,
        groups_per_pack,
        weight_stride_0,
        weight_stride_1,
        indices_stride_0,
        indices_stride_1,
        pack_weights_stride_0,
        pack_weights_stride_1,
        pack_index_stride_0,
        pack_index_stride_1,
        rank_in_pack_stride_0,
        rank_in_pack_stride_1,
        BLOCK_PACKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_PACKS)
        mask = offs < num_packs

        pack_weights_ptrs = (
            pack_weights_ptr + pid * pack_weights_stride_0 + offs * pack_weights_stride_1
        )
        pack_items_ptrs = (
            pack_items_ptr + pid * pack_weights_stride_0 + offs * pack_weights_stride_1
        )
        pack_weights = tl.load(pack_weights_ptrs, mask=mask, other=0.0)
        pack_items = tl.load(pack_items_ptrs, mask=mask, other=groups_per_pack)
        candidate_pack_weights = tl.where(
            pack_items < groups_per_pack, pack_weights, float("inf")
        )
        chosen_pack = tl.argmin(candidate_pack_weights, axis=0)
        chosen_rank = tl.sum(
            tl.where(offs == chosen_pack, pack_items, 0),
            axis=0,
        )

        group = tl.load(indices_ptr + pid * indices_stride_0 + step * indices_stride_1)
        tl.store(pack_index_ptr + pid * pack_index_stride_0 + group * pack_index_stride_1, chosen_pack)
        tl.store(
            rank_in_pack_ptr + pid * rank_in_pack_stride_0 + group * rank_in_pack_stride_1,
            chosen_rank,
        )

        weight_val = tl.load(weight_ptr + pid * weight_stride_0 + group * weight_stride_1)
        updated_pack_weights = pack_weights + tl.where(offs == chosen_pack, weight_val, 0.0)
        updated_pack_items = pack_items + tl.where(offs == chosen_pack, 1, 0)
        tl.store(pack_weights_ptrs, updated_pack_weights, mask=mask)
        tl.store(pack_items_ptrs, updated_pack_items, mask=mask)


    @triton.jit
    def _replicate_experts_step_triton(
        weight_ptr,
        logcnt_ptr,
        phy2log_ptr,
        rank_ptr,
        expert_index,
        num_log,
        weight_stride_0,
        weight_stride_1,
        logcnt_stride_0,
        logcnt_stride_1,
        phy2log_stride_0,
        phy2log_stride_1,
        rank_stride_0,
        rank_stride_1,
        BLOCK_LOG: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_LOG)
        mask = offs < num_log

        weight_ptrs = weight_ptr + pid * weight_stride_0 + offs * weight_stride_1
        logcnt_ptrs = logcnt_ptr + pid * logcnt_stride_0 + offs * logcnt_stride_1
        weight = tl.load(weight_ptrs, mask=mask, other=float("-inf"))
        logcnt = tl.load(logcnt_ptrs, mask=mask, other=1)
        score = weight / logcnt
        chosen_log = tl.argmax(score, axis=0)
        chosen_rank = tl.sum(
            tl.where(offs == chosen_log, logcnt, 0),
            axis=0,
        )

        tl.store(phy2log_ptr + pid * phy2log_stride_0 + expert_index * phy2log_stride_1, chosen_log)
        tl.store(rank_ptr + pid * rank_stride_0 + expert_index * rank_stride_1, chosen_rank)

        updated_logcnt = logcnt + tl.where(offs == chosen_log, 1, 0)
        tl.store(logcnt_ptrs, updated_logcnt, mask=mask)

def _balanced_packing_gpu_impl(
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

    device = weight.device
    indices = weight.float().sort(-1, descending=True).indices
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device=device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    arange_layers = torch.arange(num_layers, dtype=torch.int64, device=device)
    pack_weights = torch.zeros(num_layers, num_packs, dtype=weight.dtype, device=device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)
    for step in range(num_groups):
        group = indices[:, step]
        has_capacity = pack_items < groups_per_pack
        candidate_pack_weights = torch.where(
            has_capacity,
            pack_weights,
            torch.full_like(pack_weights, float("inf")),
        )
        pack = candidate_pack_weights.argmin(dim=-1)
        rank = pack_items[arange_layers, pack]
        pack_index[arange_layers, group] = pack
        rank_in_pack[arange_layers, group] = rank
        pack_weights[arange_layers, pack] += weight[arange_layers, group]
        pack_items[arange_layers, pack] += 1
    return pack_index, rank_in_pack


def balanced_packing_gpu(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if triton is None or weight.device.type != "cuda":
        return _balanced_packing_gpu_impl(weight, num_packs)

    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        return _balanced_packing_gpu_impl(weight, num_packs)

    indices = weight.float().sort(-1, descending=True).indices.contiguous()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    pack_weights = torch.zeros(
        (num_layers, num_packs), dtype=torch.float32, device=weight.device
    )
    pack_items = torch.zeros(
        (num_layers, num_packs), dtype=torch.int32, device=weight.device
    )
    block_packs = triton.next_power_of_2(num_packs)

    for step in range(num_groups):
        _balanced_packing_step_triton[(num_layers,)](
            weight,
            indices,
            pack_weights,
            pack_items,
            pack_index,
            rank_in_pack,
            step,
            num_packs,
            groups_per_pack,
            weight.stride(0),
            weight.stride(1),
            indices.stride(0),
            indices.stride(1),
            pack_weights.stride(0),
            pack_weights.stride(1),
            pack_index.stride(0),
            pack_index.stride(1),
            rank_in_pack.stride(0),
            rank_in_pack.stride(1),
            BLOCK_PACKS=block_packs,
        )

    return pack_index, rank_in_pack


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return balanced_packing_gpu(weight, num_packs)


def _replicate_experts_impl(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or weight.device.type != "cuda":
        return _replicate_experts_impl(weight, num_phy)

    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0

    phy2log = torch.arange(num_phy, dtype=torch.int64, device=weight.device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=weight.device)
    logcnt = torch.ones(n, num_log, dtype=torch.int32, device=weight.device)
    block_log = triton.next_power_of_2(num_log)

    for expert_index in range(num_log, num_phy):
        _replicate_experts_step_triton[(n,)](
            weight,
            logcnt,
            phy2log,
            rank,
            expert_index,
            num_log,
            weight.stride(0),
            weight.stride(1),
            logcnt.stride(0),
            logcnt.stride(1),
            phy2log.stride(0),
            phy2log.stride(1),
            rank.stride(0),
            rank.stride(1),
            BLOCK_LOG=block_log,
        )

    return phy2log, rank, logcnt.to(torch.int64)


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    balanced_packing_fn=balanced_packing,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
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

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing_fn(
        tokens_per_group, num_nodes
    )
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing_fn(
        tokens_per_phy, num_gpus // num_nodes
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
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


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    num_layers, num_logical_experts = weight.shape
    if enable_hierarchical:
        weight = weight.float().cpu()
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
            balanced_packing_fn=balanced_packing_cpu,
        )
    else:
        weight = weight.float()
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = num_replicas - num_logical_experts + 1
    log2phy: torch.Tensor = torch.full(
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


__all__ = ["rebalance_experts"]
