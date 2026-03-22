# EPLB Async 改为 copy 完成后立即 publish layer metadata

## 背景

- 当前 async 路径里，layer 的 H2D copy 在 `forward N` 内逐层完成。
- 上一版最小改动把 metadata 提交时机从 `forward end` 挪到了 `forward start`，但这仍然保留了两个问题：
  - routing 可见状态和统计解释状态仍耦合在同一个全局 metadata 上
  - async 统计仍然依赖 dump 时按全局 metadata 把 physical count 解释回 logical count
- 这会导致无法像 TRT-LLM 一样在某个 layer 的 H2D 完成后立刻提交该 layer 的新 placement；否则旧统计窗口会被新 mapping 解释错。

## 本次修改

- 保留 `PublishedExpertLocationMetadata` 这套稳定对象身份和 per-layer publish 框架，不额外引入新的 routing manager。
- 把 async 统计改成“事件发生时就累计 logical count”，并支持按 layer 清零：
  - `select_experts` 路径：在 logical `topk_ids` 上直接累计 logical count
  - `deepep` / `mooncake` dispatcher 路径：在 dispatch 时立即按当前 layer mapping 把 physical count 转成 logical count
- `dump_record(output_mode="object")` 在 async 下直接返回 all-reduce 后的 logical count，不再依赖 dump 时的全局 metadata 转换。
- metadata publish 时机改成：
  - layer 的 H2D 完成后，completion 线程立即 `publish_layers_from(new_metadata, [layer_id])`
  - 紧接着对该 layer 执行 `reset_async_layer_statistics([layer_id])`
  - 然后再把该 layer signal re-arm 到 GPU stage
- `copy_pairs == 0` 的 layer 也走同样的“立即 publish + reset + rearm”语义。

## 目的

- 对齐 TRT-LLM 的核心语义：
  - 某个 layer rebalance 完成并 commit placement 后，该 layer 立即开始新的统计周期
  - 后续 routing 可以立刻看到该 layer 的新 placement，而不必等下一个 `forward start`
- 消除“旧统计窗口被新 metadata 解释”的精度风险。
- 继续避免 `forward end` 上的大块 metadata merge 阻塞 host 调度。

## 约束

- 当前仍然复用 SGLang 现有 recorder 和 `PublishedExpertLocationMetadata`，没有完全照搬 TRT-LLM 的独立 routing state。
- 这次改动的成立前提是：async 统计已经改成按事件直接累计 logical count，并允许在 layer commit 时单层清零。
