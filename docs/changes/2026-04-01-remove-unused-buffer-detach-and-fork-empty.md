# 删除旧 async snapshot 遗留的无效 buffer 接口

这次修改继续清理旧 async snapshot 路径留下来的无效代码，只删除已经没有调用的 buffer 辅助方法，不改变当前 recorder 主流程。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中不再使用的方法：
  - `_StatAccumulator.detach_global_physical_count_buffer()`
  - `_StatAccumulator.detach_physical_to_logical_map_buffer()`
  - `_Buffer.fork_empty()`
  - `_CircularBuffer.fork_empty()`
  - `_InfiniteBuffer.fork_empty()`

## 变更原因

- 这些方法原本服务于旧的 async snapshot / detach buffer 方案。
- 当前 async 主流程已经改为直接 `dump_record(output_mode="object")`，不再走 detach + fork 新 buffer 的路径。
- 继续保留这些接口会增加理解成本，也会让人误以为当前仍然存在一套独立的 snapshot buffer 切换逻辑。

## 结果

- `expert_distribution` 里的 buffer 接口更贴近当前实际使用路径。
- 删除了不会再被调用的辅助方法，减少无效代码。
