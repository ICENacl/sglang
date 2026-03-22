# EPLB Async 对齐 TRT-LLM Signal / Runtime 边界

## 本次改动

- 新增 runtime 自管的 signal ABI 与 CUDA kernel：
  - `python/sglang/jit_kernel/csrc/eplb_async_signal_runtime.h`
  - `python/sglang/jit_kernel/csrc/eplb_async_signal_runtime.cu`
- `copy stream` 改为显式 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`，不再依赖 ATen stream pool 的隐式语义，对齐 TRT-LLM 的 update stream 创建方式。
- `eplb_async_runtime_cpp` 负责：
  - layer 注册时自建 host-visible signal
  - `prepare_capture_step(step)`
  - `start_iter(step, enable_statistic)`
  - 当前流上的 `wait_gpu_stage(layer_id)` / `set_cpu_stage(layer_id)`
  - 后台 `worker_loop + update_loop`
- `ExpertLocationUpdater` 收敛成薄封装：
  - 删除 Python signal JIT 依赖
  - 删除 `aux stream + event bridge`
  - 删除 Python 侧 signal tensor / encode / decode / host-register 逻辑

## 对齐到 TRT-LLM 的部分

- signal 位布局改为：
  - bit 0: owner
  - bit 1: `skip_step`
  - bits 2-62: `step`
  - bit 63: `disabled`
- GPU 侧 signal op 直接落在当前 compute stream，而不是走旁路 aux stream。
- host worker 每轮按 layer 执行：
  - `wait_last_update_done`
  - `set_gpu_stage_host`
  - `wait_cpu_stage_host`
  - `enqueue_update`
- `start_cpu_new_iter()` 会像 TRT-LLM 一样，把 `iter_id / statistic_enabled / update_enabled` 记到 per-layer 状态里。
- 只有本轮真正命中 active plan 的 layer 才会被标成 `update_enabled=true` 并进入 update queue；其他 layer 仍走 signal 协议，但不会被当成可更新层。
- update 完成后不再做“同一步 re-arm”；下一次 `start_iter()` 才重新设置 `GPU owner`。

## 保留 SGLang 差异的部分

- 数据面仍然是 `host mirror -> H2D -> live weight`。
- 没有切到 TRT-LLM 的 host-accessible live weight。
- 当前 `ExpertDistributionRecorder` 没有在这次改动里重写成 TRT-LLM 的 device-side `enabled` 消费链。
- `dump_record()` 仍然保留在 SGLang 里；与 TRT-LLM 不同，它还需要在 rebalance 边界汇总 logical count。
- 作为边界对齐的最小修改，SGLang 的 async/sync `dump_record` 现在显式复用已有 `moe ep device group` 做 `all_reduce`，不再依赖隐式默认 process group 或 world group。

## pre-EPLB sync 约束

- 本次重构显式避免在 EPLB 真正开始前新增以下同步：
  - Python `current_stream.wait_event(...)`
  - Python aux stream bridge
  - `cudaStreamWaitEvent`
  - `cudaEventSynchronize`
  - `torch.cuda.synchronize()`
- 允许保留的同步点只在后台 update 线程的 copy 完成路径里，不回流到 forward start 或 layer 入口前。
- 使用显式 non-blocking copy stream 的目的，是避免 update stream 和主图流之间出现 default-stream/stream-pool 语义不清导致的隐式串行。

## 2026-03-19 debug 开关
