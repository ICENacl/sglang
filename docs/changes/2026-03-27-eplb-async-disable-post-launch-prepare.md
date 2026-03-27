# EPLB Async 关闭 post-launch prepare，恢复到 d98 正确时序

## 背景

已知基线表明：

- `d98fc991c3cfb44e16b2f5072f0fc5e6037c6b7f` 的 async EPLB 精度正确
- 后续 `089dceafa` 引入的 `post-launch prepare` 会带来精度问题

虽然之后又补了统计路径修正，但当前主线在精度上仍然不稳定。

## 修改

- `EPLBManager._should_use_post_launch_async_prepare()` 直接返回 `False`
- async EPLB 恢复到 `d98` 的 prepare/apply 时序：
  - `forward N start` 做 prepare
  - `forward N end` 提交 apply
  - `forward N+1` 开始实际消费 async plan

## 结果

- 不再走 `089dceafa` 引入的 post-launch prepare 路径
- async EPLB 优先恢复 correctness，避免继续引入精度回退
