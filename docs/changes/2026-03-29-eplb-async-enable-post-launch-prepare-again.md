# EPLB Async 重新启用 post-launch prepare

这次修改重新打开 `post-launch async prepare`，但保持当前已经对齐好的 `logical count` 统计语义不变。

具体行为：

- `forward N-1 end` 只创建 rebalance 输入快照，不在 forward 结束时同步执行完整 prepare。
- `forward N` 的 graph launched 后，后台线程继续完成：
  - `physical count + frozen mapping` 转 `logical count`
  - `logical count` 拷到 CPU
  - `ExpertLocationMetadata.init_by_eplb(...)`
- prepare 完成后，在 `forward N` 或更晚的 `forward` 结束时 apply。
- 这次修改不恢复 boundary skip，也不恢复 layer reset；因此 async/sync 的 `logical count` 语义继续保持一致。
