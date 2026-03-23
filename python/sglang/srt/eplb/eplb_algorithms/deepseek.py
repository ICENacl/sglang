# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Tuple

import torch

from . import deepseek_vec


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
    num_local_physical_experts = num_replicas // num_gpus
    tokens_per_expert = weight.unsqueeze(0)

    phy2log, log2phy, logcnt = deepseek_vec.rebalance_experts(
        tokens_per_expert=tokens_per_expert,
        num_physical_experts=num_replicas,
        num_local_physical_experts=num_local_physical_experts,
        num_groups=num_groups,
        num_nodes=num_nodes,
        enable_hierarchical=enable_hierarchical,
    )
    return (
        phy2log.to(torch.int64),
        log2phy.to(torch.int64),
        logcnt.to(torch.int64),
    )


__all__ = ["rebalance_experts"]
