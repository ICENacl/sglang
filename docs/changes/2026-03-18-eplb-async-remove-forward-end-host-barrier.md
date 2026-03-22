# EPLB Async 去除 forward end 的 host barrier

## 背景

- 当前 `ExpertLocationUpdater.on_forward_pass_end()` 会等待 active plan 中所有 `LAUNCHED` layer 清零。
- 这会把：
  - H2D 完成
  - metadata update
  - stats reset
  - signal rearm
  的收尾路径，强行串到 host scheduler 的 `forward end` 上。
- 在 nsys 中表现为：
  - graph 之间出现 host 侧 bubble
  - scheduler 被额外拖慢约数毫秒

## 本次修改

- 删除 [expert_location_updater.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py) 中 `on_forward_pass_end()` 对 `LAUNCHED` 状态清零的 `wait_for(...)`。
- 保留 debug 日志，但改成只观测：
  - `PUBLISHED`
  - `LAUNCHED`
  - `PENDING`
  的当前分布，不再阻塞 host。
- 新增 `_maybe_finalize_active_forward_async_plan()`：
  - 在 `forward start` 前尝试回收已经完全结束的 active plan
  - 在 layer publish 完成后再次尝试回收
- active plan 的生命周期不再依赖 `forward end` 的 host barrier，而由异步 completion 路径自然推进。

## 结果

- host scheduler 不再为“已启动 H2D 的 layer 收尾”停在 `forward end`。
- 保序仍然依赖 layer 入口的设备侧 `wait_gpu_stage(...)`。
- `set_gpu_stage(...)` 仍然只会在：
  - H2D 完成
  - metadata update 完成
  - stats reset 完成
  之后发生，因此不会放松正确性约束。
