# EPLB Async Direct H2D 保序设计记录

## 结论

- 当前 async 路径继续复用 SGLang 原本的 rebalance 统计入口：
  - `forward N-1` 结束时 `dump_record() -> all_reduce -> init_by_eplb(...)`
  - 这一步只为 `forward N` 准备更新计划，不在 `forward N-1` 结束时直接 launch H2D。
- `forward N` 开始时激活上一轮已经准备好的 rebalance plan。
- `forward N` 内，某个 layer 在 `set_cpu_stage()` 后即可立刻启动该 layer 的 H2D 权重交换，不再等整个 `forward N` 结束。
- async 统计重新对齐到 sync 语义：窗口内继续累计 `physical count`，并按 forward 冻结 `physical_to_logical_map`，在 rebalance 边界统一转换 `logical count`。
- 因此新的 `expert_location_metadata` 可以在某个 layer 的 H2D 完成后立即按 layer 发布，不再等到 `forward N+1` 开始前。
- 如果当前 plan 跨多个 split/piecewise forward 才能覆盖完所有 layer，则 metadata 按“本轮已经完成 copy 的 layer slice”渐进提交，未执行到的 layer 保留到后续 forward 继续完成。
- 因此当前实现的时序是：
  - `forward N-1` 产出 plan
  - `forward N` 执行 copy
  - `forward N` 内某个 layer copy 完成后立即 publish 该 layer metadata
  - 后续 forward 从该 layer 下一次真正参与路由起按新 metadata 继续累计

## 为什么放在 EP 子类

- `eplb async` 只服务于 EP 路径，当前实际入口是 `DeepEPMoE.forward()`。
- `DeepEPMoE` 自身已经重写 `forward()`；如果继续把 async 生命周期放在通用 `FusedMoE` 基类，会让真实执行路径和同步点分离。
- 因此保留通用基类里的 `expert_location_metadata` 能力，但把 async 的“层入口提交”收敛到 EP 子类。

## 当前协议

### 1. rebalance 阶段

- `ExpertLocationUpdater.update()` 仍然负责根据新旧 `physical_to_logical_map` 计算哪些本地 physical slot 需要更新。
- async 模式下，不再在 layer 入口执行 graph 内权重提交。
- rebalance 只构造每个 layer 的 H2D 计划：`[(logical_expert_id, dst_slot)]`。
- H2D 源统一来自 async host mirror；same-gpu/free-rider 在 async 路径上不再复用 live 权重，优先保证同步语义和 cudagraph 安全。

### 2. layer 入口

- `DeepEPMoE.forward()` 入口先执行 `wait_gpu_stage(self.layer_id)`。
- `wait_gpu_stage()` 现在直接转发到 `eplb_async_runtime_cpp.wait_gpu_stage(layer_id)`。
- runtime 在当前 layer 的 compute stream 上直接发射 device-side wait kernel，并写入每层预分配的 `enabled` tensor。
- 这里不再使用 Python `aux stream + event bridge`，也不再在 layer 入口前插入额外的 cross-stream wait。
- 因此 `ModelRunner.on_forward_pass_start()` 到第一个 layer `wait_gpu_stage()` 之间，不会因为 EPLB signal 链路新增 stream sync。

### 3. layer 出口

- `DeepEPMoE.forward()` 在 `finally` 中执行 `set_cpu_stage(self.layer_id)`。
- `set_cpu_stage()` 现在直接转发到 `eplb_async_runtime_cpp.set_cpu_stage(layer_id)`。
- runtime 在当前 compute stream 上直接发射 device-side set kernel，把该 layer 的 signal 切到 `CPU owner`。
- 这样无论 forward 正常返回还是抛异常，CPU worker 都能在 host 侧看到“该 layer 已退出本轮计算”。

### 4. forward 内的图外 H2D

- `EPLBManager.rebalance()` 在 `forward N-1` 结束时只准备下一轮 plan：
  - 计算新的 `ExpertLocationMetadata`
  - 为每个目标 layer 构造 `copy_pairs`
  - 把 plan 交给 `ExpertLocationUpdater`
- `ModelRunner.forward()` 开始时调用 `expert_location_updater.on_forward_pass_start()` 激活这份 plan。
- 当前实现不再依赖 Python `set_cpu_stage()` 直接消费 plan。
- `DeepEPMoE.forward()` 的 `set_cpu_stage(self.layer_id)` 只负责在 graph 内把 signal 切到 `CPU owner`。
- `ExpertLocationUpdater` 不再维护 Python monitor/copy worker，也不再持有 per-layer signal tensor。
- plan 消费、host `wait_cpu_stage`、copy stream 上的 H2D、GPU metadata publish 与 CPU metadata mirror 都收敛到 `eplb_async_runtime_cpp`：
  - `start_iter(step, enable_statistic)` 只入队 iter，不在 forward 前半段等待设备事件
  - `worker_loop()` 按 layer 执行 `wait_last_update_done -> set_gpu_stage_host -> wait_cpu_stage_host -> enqueue_update`
  - `update_loop()` 只负责 H2D/update task
- 这样 H2D 的真实启动点由“设备侧 signal 被 host 观察到”决定，而不是 replay 时是否执行到 Python hook。

### 5. metadata publish 与统计窗口

- 当前 forward 内，同一个 layer 在 `set_cpu_stage()` 之后才允许 host 启动该 layer 的 H2D，因此本轮该 layer 的 compute 与该次更新不会再并发读写同一 live slot。
- 为了让 metadata 更新不污染 rebalance 输入，async 统计继续走“先累计 physical count，rebalance 时再统一转 logical”的链路，但为每个 forward 冻结一份 `physical_to_logical_map`：
  - `select_experts` 路径：累计 physical expert 命中数
  - `deepep` / `mooncake` dispatcher 路径：累计 dispatch 侧 physical count
  - rebalance 时按每个 forward 自己冻结的 mapping 做 `scatter_add`
- 当某个 layer 的 H2D 完成后，completion 线程立刻执行：
  - `current_metadata.update(new_metadata, update_layer_ids=[layer_id])`
- 不再清零该 layer 的统计窗口；旧 forward 的统计仍按各自冻结的 mapping 解释
- 当前实现不再在 update 完成时把同一步 signal re-arm 回 `GPU owner`。
- 下一次真正的 `start_iter(step)` 才会由 host 写入新的 `GPU owner + step/skip_step` signal。
- 这和 TRT-LLM 的边界一致，避免 update 线程在同一步里反向改写 signal。
- 这样即使存在 split/piecewise forward：
  - 已完成 copy 的 layer 会立即切到新 metadata
  - 旧 forward 的统计继续保留，并按各自冻结的 mapping 解释
  - 未执行到的 pending layer 继续保留在 active plan 中，滚到后续 forward 再完成
  - `forward end` 仍只等待本轮已启动但尚未完成的 layer copy 收尾；未启动的 pending layer 不阻塞本轮结束

### 6. replay 语义

- capture 时录进去的是 graph 内的 `wait_gpu_stage` / `set_cpu_stage` signal op，不再等待图外旧 event。
- replay 时，GPU 侧只会等待 signal 变成 GPU owner；真正的权重更新仍由图外 copy stream 完成。
- 因为 host 只有在 copy 完成后才会 re-arm GPU owner，所以 replay 阶段在 layer 首次读权重前仍具备显式等待点。
- capture/warmup 阶段只做 capture-only signal 预备：
  - 重置 per-layer signal 到 GPU owner
  - 不激活真实 async plan
  - 不推进真实 rebalance 计数
  - 不提交 metadata
- 因此“真实 forward 生命周期”和“graph capture/warmup 生命周期”在实现上已经分离。

## 为什么这条路能避开 cudagraph 问题

- `StreamCaptureIsolation` 的根因是 capture 流去依赖 capture 外其它 stream 上最后一次 record 的 event。
- 新实现里，capture 只看到当前 compute stream 上的 signal wait/set kernel。
- Python 侧不再创建 aux stream，也不再通过 event bridge 把 signal op 从主流旁路出去。
- producer 和 consumer 都属于同一次 capture / replay 的当前流，因此不会再出现“graph wait 图外旧 event”的问题。

## 与旧方案的差异

- 删除 `weights_ready` / `main_done` 的 per-layer 事件状态机。
- 删除 `pending commit bank + graph 内 commit kernel` 方案。
- 删除 `sgl-kernel` 扩展里的 `eplb_commit_pending_tensor` / `eplb_commit_finalize`。
- graph 内只保留 runtime 当前流上的轻量 signal kernel：
  - `eplb_async_runtime_cpp.wait_gpu_stage`
  - `eplb_async_runtime_cpp.set_cpu_stage`
- `set_gpu_stage` 改为 runtime host 侧写 signal，不再由 Python 管理 signal tensor。
- async 更新源统一收敛为 `host mirror -> live 权重` 的 H2D memcpy。
- 当前 async 路径不保留 same-gpu/free-rider live->live 复用优化。
- 当前 async plan 的消费不再依赖 `DeepEPMoE.forward()` 中的 Python launch hook，而是依赖 runtime host worker 观察设备侧 signal。
- async metadata publish 不再调用 `ExpertLocationMetadata.update()` 做整表 merge，而是保持全局 metadata 对象稳定，在对象内部做 per-layer published view 切换。
- async 统计恢复为 sync 对齐语义：按 forward 冻结 mapping，再在 rebalance 边界统一把 buffered physical count 转成 logical count。

## 代码落点

- Python 状态管理：
  - `python/sglang/srt/eplb/expert_location_updater.py`
  - `python/sglang/srt/layers/moe/ep_moe/layer.py`
  - `python/sglang/srt/model_executor/model_runner.py`
- C++ runtime + signal kernel：
  - `python/sglang/jit_kernel/csrc/eplb_async_runtime.cpp`
  - `python/sglang/jit_kernel/csrc/eplb_async_signal_runtime.h`
  - `python/sglang/jit_kernel/csrc/eplb_async_signal_runtime.cu`
- runtime 自己分配并管理 host-visible signal，不再依赖 Python JIT signal module 的预热或 aux stream bridge。

## 已知约束

- 该方案默认不引入 recapture。
- 当前环境没有 `torch` 运行时，代码只做了静态改动和 Python 语法检查；真实的 cudagraph capture / replay 行为仍需在目标 CUDA 环境验证。
- 旧的 `stream-ordered event` 排查结论和 `pending commit bank` 尝试方案已合并进本设计记录，不再作为实现基线。

## 2026-03-18 控制面重构

- 当前实现继续沿 TRT-LLM 的 `workerThread + computeAndUpdateThread` 思路收敛，但保持 SGLang 现有 `host mirror -> H2D -> live weight` 数据路径。
- Python `ExpertLocationUpdater` 不再维护 monitor/copy completion 线程；forward 主路径只负责：
  - `prepare_async_layers()` 注册 layer
  - `submit_prepared_async_plan()` 提交 rebalance plan
  - `on_forward_pass_start()` 提交 iter step
  - graph 内 `wait_gpu_stage()` / `set_cpu_stage()`
- 后台控制面迁移到新的 `eplb_async_runtime_cpp`：
  - host 侧 `set_gpu_stage`
  - host 侧 `wait_cpu_stage`
  - update task 调度
  - copy stream 上的 H2D、GPU metadata 切换
  - update 完成后的 CPU metadata mirror 更新
- C++ 源文件放在 `python/sglang/jit_kernel/csrc/eplb_async_runtime.cpp`，与现有 EPLB signal JIT kernel 的代码布局保持一致。
- 这次重构的目标不是改数据路径，而是把原来会拖慢 host scheduler 的 Python 轮询和 Python publish 状态机移出主路径。

## 2026-03-19 TRT-LLM 边界对齐补充

- signal ABI 改成 TRT-LLM 风格：
  - bit 0: owner
  - bit 1: `skip_step`
  - bits 2-62: `step`
  - bit 63: `disabled`
- `prepare_capture_step(step)` 只做 host signal 预置，不激活真实 plan，也不引入 stream/event 同步。
- `wait_gpu_stage()` / `set_cpu_stage()` 直接在当前 compute stream 上发射 kernel。
- 为避免引入新的 pre-EPLB bubble，本次实现明确不在以下路径增加新的 `cudaStreamWaitEvent` / `cudaEventSynchronize`：
  - `prepare_async_layers()`
  - `on_forward_pass_start()`
  - `on_capture_start()`
  - layer 入口 `wait_gpu_stage()` 之前
- 当前仍保留 update 线程里 copy 完成后的后台 `cudaEventSynchronize`，但它不会回流到 forward 起点或 layer 入口前。
