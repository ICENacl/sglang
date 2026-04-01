# 删除 async 专用的 global physical count 统计链

这次修改继续把 `expert_distribution` 中 async 的统计路径收敛到和 sync 一致，不再为 async 单独维护 `global_physical_count`。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中 async 专用的状态和逻辑：
  - `_async_physical_count_source`
  - `_async_global_physical_count`
  - `_append_async_from_topk_ids()`
  - `_append_async_from_local_physical_count_list()`
  - `_append_async_from_local_physical_count_tensor()`
  - `single_pass_data["async_global_physical_count"]`
- `metric_global_physical_count` 的统计来源保持不变，但不再依赖 `_compute_async_physical_count_source()`，而是直接使用同样的来源判断逻辑。
- `_StatAccumulator.append()` 不再区分 async/sync 的 `global_physical_count` 来源，统一使用 `single_pass_data["global_physical_count"]`。

## 变更原因

- 当前希望 async 和 sync 在 `logical_count` 的统计来源上保持一致。
- 之前 async 仍然额外维护了一套 `async_global_physical_count`，使 async 的统计链和 sync 不同。
- 这次修改将这套 async 专用计数链完全删除。

## 结果

- async 和 sync 现在统一使用 `global_physical_count -> logical_count` 这条主路径。
- `expert_distribution` 中与 async 专用 count 相关的额外状态和分支进一步减少。
