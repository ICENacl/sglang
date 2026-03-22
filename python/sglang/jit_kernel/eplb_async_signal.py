from __future__ import annotations

import logging
from functools import lru_cache

import torch

from sglang.jit_kernel.utils import load_jit

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _jit_eplb_async_signal_module():
    return load_jit(
        "eplb_async_signal",
        cuda_files=["eplb_async_signal.cuh"],
        cuda_wrappers=[
            ("eplb_wait_gpu_stage", "eplb_wait_gpu_stage"),
            ("eplb_set_cpu_stage", "eplb_set_cpu_stage"),
        ],
    )


def warmup_eplb_async_signal_module() -> None:
    _jit_eplb_async_signal_module()


def eplb_wait_gpu_stage(
    signal_step_and_owner: torch.Tensor, stream_tensor: torch.Tensor
) -> None:
    module = _jit_eplb_async_signal_module()
    module.eplb_wait_gpu_stage(signal_step_and_owner, stream_tensor)


def eplb_set_cpu_stage(
    signal_step_and_owner: torch.Tensor, stream_tensor: torch.Tensor
) -> None:
    module = _jit_eplb_async_signal_module()
    module.eplb_set_cpu_stage(signal_step_and_owner, stream_tensor)
