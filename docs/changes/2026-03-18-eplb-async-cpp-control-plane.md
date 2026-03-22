# EPLB Async 控制面下沉到 C++ Runtime

## 背景

原有 SGLang EPLB async 控制逻辑主要依赖 `ExpertLocationUpdater` 中的 Python 线程与 host 轮询：

- `on_forward_pass_start()` 在 host 主路径重置所有 layer 的 signal
- Python monitor 线程轮询 `signal_step_and_owner.item()`
- Python worker 线程等待 copy 完成并串行执行 metadata publish

这会让 EPLB async 的控制面和 host scheduler 竞争 CPU/GIL，并在 graph 之间引入 bubble。

## 本次改动

本次实现沿 TRT-LLM 的 `workerThread + computeAndUpdateThread` 思路重构：

1. 新增 `eplb_async_runtime_cpp` C++ runtime。
   源文件放在 `python/sglang/jit_kernel/csrc/eplb_async_runtime.cpp`，signal kernel 在：
   - `python/sglang/jit_kernel/csrc/eplb_async_signal_runtime.h`
   - `python/sglang/jit_kernel/csrc/eplb_async_signal_runtime.cu`
2. 将以下控制面职责下沉到 C++ 后台线程：
   - per-iter `setGpuStage`
   - `waitCpuStage`
   - per-layer update task 入队
   - copy stream 上的 H2D、GPU metadata 更新、统计清零
   - 更新完成后的 CPU metadata 切换
3. Python `ExpertLocationUpdater` 收敛为薄封装，只负责：
   - 注册 layer
   - 构造 async plan
   - 在 forward start 提交当前 iter
   - graph 内 `wait_gpu_stage` / `set_cpu_stage`
4. `wait_gpu_stage` / `set_cpu_stage` 不再走 Python `aux stream + event bridge`，而是直接在当前 compute stream 上发射 runtime signal kernel。
5. signal ABI 对齐 TRT-LLM：
   - bit 0: owner
   - bit 1: `skip_step`
   - bits 2-62: `step`
   - bit 63: `disabled`

## 当前语义

- `forward N` 中 layer x 完成后，后台线程观察到 `CPU stage`，即可启动该 layer 的 H2D。
- `forward N+1` 的 layer x 在 `wait_gpu_stage()` 处等待新权重 ready 后再继续。
- 每个 layer 完成 H2D 后，会立刻发布该 layer 的 metadata，并清空该 layer 的 async 统计窗口。
- update 完成后不再把同一步 signal re-arm 回 `GPU owner`；下一次 `start_iter()` 才会为下一轮设置新的 signal。
- worker 线程对每层显式维护 `iter_id / statistic_enabled / update_enabled`，只有命中当前 plan 的 layer 才会被视为本轮可更新层。

## 与 TRT-LLM 的对应关系

- `worker_loop()` 对应 TRT-LLM 的 `workerThread()`
- `update_loop()` 对应 TRT-LLM 的 `computeAndUpdateThread()`
- 继续保留 SGLang 当前 `host mirror -> H2D -> live weight` 的数据路径，不切换到 TRT-LLM 的 host-accessible live weight 路线

## 后续验证重点

1. nsys 中 host scheduler 是否不再停在 Python monitor/copy worker 上。
2. EPLB 开始前是否不再引入新的 cross-stream wait / host-blocking sync。
3. H2D 是否仍能与后续 layer 计算 overlap。
4. update stream 是否保持为显式 non-blocking stream，并且只在后台 update 路径里做完成确认。

## 2026-03-18 补充修复

- 修复 `python/sglang/srt/eplb/expert_location_updater.py` 中 `_update_expert_weights_raw()` 的缩进错误。
- 该错误会导致 async 路径下 `layer_plans` 没有真正构建，最终在 `submit_prepared_async_plan()` 中按 `update_layer_ids` 取 plan 时触发 `KeyError`。
- 同时补充了防御性报错，便于后续直接定位 `update_layer_ids` 和 `layer_plan_keys` 不一致的问题。
