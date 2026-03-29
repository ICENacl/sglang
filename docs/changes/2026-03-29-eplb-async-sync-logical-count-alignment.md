# EPLB Async 统计语义对齐 Sync

这次修改把 `eplb async` 的 rebalance 统计语义重新对齐到 `eplb sync`：

- async 继续使用 `physical count + per-forward frozen mapping snapshot` 生成 `logical count`
- async 不再在 layer metadata/update 完成后清零该 layer 的统计窗口
- async 不再在 rebalance 边界额外跳过下一轮 forward，统计窗口长度与 sync 保持一致
- async 和 sync 的区别只保留在 prepare/apply 的时序控制
