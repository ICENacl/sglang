# EPLB Async 修复分段 forward 下的 metadata 提交精度问题

## 问题

上一版 async 实现把一个 rebalance plan 视为“单个 `ModelRunner.forward()` 内一定能覆盖完的更新集合”。

这在 `split_prefill`、piecewise forward 等分段执行场景下并不成立：

- 某次 forward 只执行了 plan 中的一部分 layer
- 但结束时却把整份 plan 的 metadata 一次性提交
- 下一次 forward 会按新 metadata 路由到还没完成换权重的 layer
- 直接导致 metadata 和 live weight 不一致，表现为精度问题

## 修复

- async plan 改为可跨多个 forward 持续存在
- 每个 plan 维护 `pending_layer_ids`
- 某个 layer 真正执行到并启动更新后，才从 `pending_layer_ids` 中移除
- `on_forward_pass_end()` 在 `dump_record()` 之后，只提交“本轮已经完成 copy 的 layer slice”
- 尚未执行到的 layer 继续保留在 active plan 中，后续 forward 再完成

## 结果

- dump 仍然复用 SGLang 原有逻辑，且不会被新 metadata 污染
- 已经完成换权重的 layer，不会在下一轮继续按旧 metadata 路由
- 未完成的 layer，也不会被过早切到新 metadata
