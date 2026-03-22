# eplb async cudagraph capture 缺失 forward-end 回调修复记录

## 改动原因

- 当前 `eplb async` 的 signal 状态机依赖每轮 forward 结束后执行 `on_forward_pass_end()`，这样才能：
  - 为没有更新的 layer 直接 re-arm 到下一 step 的 GPU owner
  - 为有更新的 layer 在 copy 完成后切回 GPU owner
- 但 `CudaGraphRunner` 和 `PiecewiseCudaGraphRunner` 在 warmup/capture 阶段直接调用的是 `model.forward(...)`，绕过了 `ModelRunner.forward()`。
- 因此 capture 期间：
  - layer 入口 `wait_gpu_stage()` 会等待 GPU owner
  - layer 出口 `set_cpu_stage()` 会把 signal 置成 CPU owner
  - 但这一轮结束后没有执行 `expert_location_updater.on_forward_pass_end()`
- warmup/capture 多跑几轮后，下一轮又进入 `wait_gpu_stage()` 时，signal 还停在 CPU owner，于是表现为 capture 过程很慢或直接 hang。

## 改动内容

1. 在 `ModelRunner` 中新增统一入口：
   - `on_forward_pass_end()`
   - 内部同时调用：
     - `self.eplb_manager.on_forward_pass_end()`
     - `self.expert_location_updater.on_forward_pass_end()`
2. 在 `CudaGraphRunner.capture_one_batch_size()` 中：
   - warmup 的两轮 `run_once()` 后调用 `self.model_runner.on_forward_pass_end()`
   - 实际 `_capture_graph(...)` 返回后再调用一次 `self.model_runner.on_forward_pass_end()`
3. 在 `PiecewiseCudaGraphRunner` 中：
   - `warmup_torch_compile()` 的 direct `model.forward(...)` 后调用一次 `self.model_runner.on_forward_pass_end()`
   - `capture_one_batch_size()` 的每轮 warmup `run_once()` 后调用一次 `self.model_runner.on_forward_pass_end()`

## 预期效果

- capture/warmup 阶段也能把 eplb async signal 状态机闭环。
- 避免 layer 在上一轮 capture 结束时停留在 CPU owner，导致下一轮 `wait_gpu_stage()` 卡住。
- 不改变 replay 路径的保序语义。
