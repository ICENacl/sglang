## EPLB Async apply/submit 时序调整

- 问题：async 路径原先会在 `init_by_eplb(...)` 生成新 placement 的同一轮 `forward end` 里立即 `apply/submit`。
- 这会让 `submit_plan` 的时机过早，和“当前轮统计只用于下一轮 expert 排布”的语义不一致。

本次修改后：

- 非 post-launch async 路径：
  - 在 rebalance 边界的 `forward end` 只做 `dump_record -> logical_count -> init_by_eplb(...)`
  - 把生成好的 placement 缓存在 manager 中
  - 到下一轮 `forward end` 再 `apply/submit`
  - 默认路径用 manager 内部的 generator 直接表达这个时序，并统一使用 `model_runner.forward_pass_id`

- post-launch async 路径：
  - 后台线程完成 `init_by_eplb(...)` 后，不会在当前轮立即 `apply`
  - manager 会把 apply 目标设为下一轮 `forward end`

现在统一语义为：

- `init_by_eplb(...)` 所在轮：只生成新 placement
- 下一轮 `forward end`：才允许 `apply/submit`
- snapshot 的 `ready_event` 只表示统计快照可读
- placement 的 `apply_event` 单独在 `init_by_eplb(...)` 完成后记录，只用于下一轮 `apply/submit` 前同步

这样 `submit_plan` 的时机和统计窗口语义一致，不再依赖 runtime 里的 `target_step` 做补偿。
