# EPLB Async 改为「N-1 产出计划，N 内按 layer 启动交换」

## 背景

上一版工作区实现把 async H2D 的真正启动点放在 `forward` 结束后的 `on_forward_pass_end()`。

这会带来两个偏差：

- `layer x` 在本轮已经完成并切回 CPU stage 后，仍然要等整次 `forward` 结束才能开始交换
- 实现语义偏离 TRT-LLM 那种“某层退出即可启动该层更新”的时序

同时，这次调整仍要求复用 SGLang 原本的统计链路，而不是改成 TRT-LLM 的 forward 内实时统计：

- `forward N-1` 结束时 dump
- `forward N` 根据 dump 结果启动 eplb 权重交换

## 本次修改

- `ExpertLocationUpdater` 的 async 路径改成两段式：
  - `on_forward_pass_start()` 激活上一轮已经准备好的 plan，并为本轮 re-arm layer signal
  - `set_cpu_stage()` 后按 layer 立即启动 H2D copy
- `on_forward_pass_end()` 不再统一 launch copy，而是只负责：
  - 等待本轮 active plan 的 copy 全部完成
  - 提交新的 `expert_location_metadata`
- `update()` 的 async 分支不再直接更新 global metadata，而是只准备“下一轮 plan”
- `ModelRunner.forward()` 在真实 forward 开始前显式调用 `expert_location_updater.on_forward_pass_start()`
- graph capture / warmup 路径只驱动 async begin/end，不推进真实 rebalance
- `ExpertDistributionRecorder` 过滤掉没有真实 forward context 的 capture hook

## 结果语义

- `forward N-1`：dump 并准备下一轮 rebalance plan
- `forward N`：layer 在退出时立即开始该层 H2D
- `forward N` 结束：等待本轮 plan 完成并提交 metadata
- `forward N+1`：开始按新的 metadata 路由并读取更新后的权重

## 影响

- 当前实现仍然保留一轮统计滞后，这是本次设计的明确约束
- 但 H2D 的启动时机已经从“整次 forward 结束”前移到“本层退出”
- 这条时序更接近 TRT-LLM 的 worker 语义，同时仍兼容 SGLang 现有 dump / rebalance 入口
