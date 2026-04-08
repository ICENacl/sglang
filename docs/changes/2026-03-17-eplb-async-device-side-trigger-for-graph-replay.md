## 背景

之前的 eplb async 实现里，plan 的消费依赖 `DeepEPMoE.forward()` 末尾的 Python 路径：

- `set_cpu_stage(layer_id)`
- `_maybe_launch_layer_update(...)`

这个方案在普通 eager forward 下可工作，但在 cudagraph replay 下存在根本问题：

- 真实 layer 计算发生在 replay 中
- 设备侧 signal op 会执行
- 但 Python hook 不会按 eager 方式重新参与每层 plan 消费

结果就是：

- active async plan 能被激活
- layer 也确实执行了
- 但 plan 长时间停留在 `pending`
- 日志表现为 `activate_async_plan ... pending_layers=[0]`，随后 `forward_end completed_layers=[]`

## 本次改动

核心目标：

- 将 async plan 的消费触发点改为 capture/replay 都成立的设备侧同步链
- 不再依赖 Python `set_cpu_stage()` 内直接 launch H2D
- 不再考虑 eager forward 语义

具体实现：

### 1. `ExpertLocationUpdater` 新增 host worker 线程

- 在真实 `forward start` 激活 active plan 后，向 worker 发布本轮 `target_step`
- worker 按 layer 观察 host-visible `signal_step_and_owner`
- 当某层 signal 变为：
  - `step == target_step`
  - `owner == CPU`
- 立即在 copy stream 上发起该层 `host mirror -> live weight` 的 H2D

### 2. `set_cpu_stage()` 只保留 signal 语义

- `DeepEPMoE.forward()` 末尾的 `set_cpu_stage(layer_id)` 仍然存在
- 但它只负责在 graph 内将 signal 切到 CPU owner
- 不再直接调用 Python plan 消费逻辑

### 3. async plan 改为显式状态机

每个 layer 改为以下状态之一：

- `pending`
- `launched`
- `done`
- `committed`

含义：

- `pending`：本轮尚未观测到 CPU stage
- `launched`：已发起 H2D，等待 copy 完成
- `done`：copy 已完成，等待 metadata commit
- `committed`：metadata 已提交

### 4. capture/warmup 生命周期与真实 forward 生命周期分离

`on_eplb_async_capture_start/end()` 改为 capture-only：

- 只重置 signal 到 GPU owner
- 不激活真实 async plan
- 不推进真实 step
- 不提交 metadata

这样可以避免 capture/warmup 把真实 plan 提前消费掉。

## 影响

预期效果：

- graph replay 下，plan 可以被真实消费
- 不再出现“layer 已执行，但 active plan 一直不前进”的假象
- `forward end` 只等待本轮已启动的 copy 收尾，不会因为未执行到的 layer 永久卡住

## 涉及文件

- `/config/workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py`
- `/config/workspace/sglang/python/sglang/srt/model_executor/model_runner.py`
- `/config/workspace/sglang/docs/developer_guide/eplb_async_direct_h2d_ordering.md`

## 验证建议

打开：

```bash
SGLANG_EPLB_ASYNC_SYNC_DEBUG=1
```

重点观察以下日志链路：

- `activate_async_plan`
- `launch_selected`
- `launch_h2d`
- `copy_done_rearm`
- `forward_end`

正确行为应表现为：

- 真实 replay 后能看到 `launch_selected/launch_h2d`
- `forward_end` 中 `completed_layers` 不再长期为空

## 说明

当前环境无法直接验证 CUDA graph replay，本次只做静态实现与 Python 语法检查。
