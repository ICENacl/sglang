# EPLB async 补充 plan 消费关键日志

本次修改只增加 async plan 消费链路上的关键日志，不改变现有执行语义。

新增日志点：

- Python 提交侧：
  - `submit_async_plan` 现在会额外打印每层 `copy_pairs` 数量
- C++ runtime：
  - `plan_queued`
  - `activate_plan`
  - `wait_cpu_stage_begin`
  - `wait_cpu_stage_end`
  - `wait_cpu_stage_disabled`
  - `enqueue_update`
  - `enqueue_update_skipped`
  - `run_update_begin`
  - `run_update_end`
  - `finish_iter`

用途：

- 区分“plan 已 submit 但未激活”
- 区分“plan 已激活但 layer 没有入队 update”
- 区分“update 已入队但没有真正开始执行”
- 区分“本层 `copy_pairs == 0`，因此没有真实 H2D”

开启方式：

- `SGLANG_EPLB_ASYNC_SYNC_DEBUG=1`
