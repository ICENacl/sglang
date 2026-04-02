# 删除 metric_global_physical_count，统一 async/sync 统计路径

这次修改删除了 `expert_distribution` 中仅供 metrics 使用的独立 `metric_global_physical_count` 计数链，让 `eplb async` 和 `eplb sync` 在统计方法上完全一致。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中与 `metric_global_physical_count` 相关的状态和逻辑：
  - `_metric_global_physical_count`
  - `_metric_source`
  - `_compute_metric_source()`
  - `_append_metric_from_topk_ids()`
  - `_append_metric_from_local_physical_count_list()`
  - `_append_metric_from_local_physical_count_tensor()`
  - `single_pass_data["metric_global_physical_count"]`

## 变更原因

- 现在希望 `eplb async` 和 `eplb sync` 使用同一套统计方法。
- 之前 `metric_global_physical_count` 只服务于 balancedness / heatmap 指标，会让 async 观测路径和主统计路径不一致。
- 删除这套独立计数后，指标和 rebalance 都统一基于同一份 `global_physical_count`。

## 结果

- async 和 sync 的统计来源完全一致。
- `expert_distribution` 中的重复计数链进一步减少。
- balancedness / heatmap 指标不再依赖单独的 metric count，而是直接基于主统计路径。
