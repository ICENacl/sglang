# 让 async 的 logical count 生成方式与 sync 保持一致

这次修改删除了 `expert_distribution` 中 async 专用的 `physical_to_logical_map` 历史缓冲，让 async 和 sync 一样，在 `dump_record()` 时统一使用当前 metadata 的 `physical_to_logical_map` 来生成 `logical_count`。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中 async 专用的辅助状态：
  - `_async_forward_physical_to_logical_map`
  - `_physical_to_logical_map_of_buffered_step`
- 删除围绕这两份状态的更新和 reset 逻辑。
- 在 `_StatAccumulator.dump()` 中，不再区分 async/sync 两条 `physical_to_logical_map` 来源。
- 现在无论 async 还是 sync，都统一使用：
  - `self._expert_location_metadata.physical_to_logical_map`

## 变更原因

- 当前希望 async 的 `logical_count` 生成方式和 sync 保持一致，减少额外的 async 专用状态和分支。
- 之前 async 会额外缓存每步的 `physical_to_logical_map` 历史，再据此生成 `logical_count`。
- 现在这套 async 专用逻辑被删除，逻辑收敛到和 sync 一样的路径。

## 结果

- `expert_distribution` 中与 async 专用 `logical_count` 生成相关的状态更少。
- async 和 sync 的 `logical_count` 生成方式保持一致。
