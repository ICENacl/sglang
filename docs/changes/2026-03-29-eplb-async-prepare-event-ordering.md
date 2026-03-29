# EPLB Async 用 prepare done event 约束 submit plan

这次修改去掉 async post-launch prepare 的代际保护，改成更直接的保序语义：

- 同一时刻只允许一份 pending/prepare/prepared rebalance 输入存在
- prepare 完成后记录 `prepared_ready_event`
- apply 进入 `update_expert_location()` 前先等待这份 event

结果是：

- `submit_plan` 只能消费当前这份 dump snapshot 计算出来、并且 prepare 真正完成了的 placement
- 不再依赖额外的 generation 判断
