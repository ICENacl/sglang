# EPLB init_by_eplb 按算法选择 CPU/GPU 统计路径

本次修改优化了 `dump_record() -> init_by_eplb()` 之间的 `logical_count` 搬运路径。

## 背景

此前 `ExpertLocationMetadata.init_by_eplb()` 会先把 `logical_count` 无条件搬到 `server_args.device`。  
但当前部分 EPLB 算法实际上仍然是 CPU 路径，例如：

- `deepseek`
- `deepseek_hierarchical`
- `deepseek_vec_hierarchical`
- `elasticity_aware`

这些算法后续又会把统计拉回 CPU，造成一次不必要的 device 往返，并把 D2H 的同步成本集中暴露在 rebalance 边界。

## 修改

- 在 `python/sglang/srt/eplb/eplb_algorithms/__init__.py` 中新增 `algorithm_runs_on_cpu()`
- 在 `python/sglang/srt/eplb/expert_location.py` 中：
  - 先计算本次 rebalance 实际采用的算法
  - CPU 算法路径下，`logical_count` 直接拷到 pinned host tensor
  - GPU 算法路径下，才继续把 `logical_count` 放到 device 上

## 预期效果

- 避免 CPU 算法路径上的“先 H2D，再 D2H”
- 将 `logical_count` 的 host 消费边界提前并显式化
- 为后续继续参考 TRT-LLM，把统计长期收敛到 host-visible 路径打基础
