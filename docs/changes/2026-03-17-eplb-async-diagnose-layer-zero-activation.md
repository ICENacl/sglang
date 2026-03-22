## 背景

在排查 eplb async 运行时行为时，日志里多次出现：

```text
[EPLBAsyncSync] activate_async_plan step=... layers=[0]
```

这条日志容易误导，因为它原先打印的是 `layer_plans.keys()`，不能明确区分以下两种情况：

1. plan 在提交阶段就只包含 `layer 0`
2. plan 原本包含多层，但当前只剩 `layer 0` 未完成

另外，也缺少足够日志来判断 layer 级 H2D 启动是否真的发生，以及 active plan 是否被消费推进。

## 本次改动

改动文件：

- `/config/workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py`

新增/调整的 debug 日志：

- `prepared_async_layers`
  - 启动后打印当前 rank 上实际参与 async 的 layer id 集合
- `submit_async_plan`
  - 打印本次提交的 `update_layer_ids` 和 `layer_plans.keys()`
- `activate_async_plan`
  - 改为同时打印：
    - `update_layers`
    - `pending_layers`
- `forward_end`
  - 打印：
    - 本轮已完成的 `completed_layers`
    - 当前剩余的 `pending_layers`
- `_maybe_launch_layer_update`
  - 对以下路径分别打点：
    - `no_active_plan`
    - `already_launched`
    - `not_in_active_plan`
    - `launch_selected`

## 预期用途

通过这组日志，可以直接判断：

1. 当前 rank 上是否真的只有 `layer 0` 参与 async
2. 提交给 async queue 的 plan 是否从源头就只有 `[0]`
3. `layer 0` 是否实际执行到了 `set_cpu_stage -> launch_h2d -> copy_done_rearm`
4. active plan 是否因为 `pending_layers` 长期不清空而卡住

## 结果判读

如果看到：

```text
prepared_async_layers layer_ids=[0]
```

那说明当前 rank 上只有一个 async layer，长期出现 `[0]` 是合理的。

如果看到：

```text
submit_async_plan update_layers=[0]
```

那说明 plan 源头就只有 `layer 0`，不是消费阶段的问题。

如果看到：

```text
activate_async_plan ... pending_layers=[0]
```

并且后续长期没有：

```text
launch_selected layer_id=0
copy_done_rearm layer_id=0
```

那说明 active plan 没有被真正消费，需要继续查 forward 路径是否经过该 layer。

## 说明

这次改动只增加诊断日志，不改变 eplb async 的执行语义。
