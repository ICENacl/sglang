# eplb async cudagraph capture JIT 预热修复记录

## 改动原因

- `hang.log` 显示主线程卡在：
  - `tvm_ffi/cpp/load_inline.py`
  - `tvm_ffi/utils/lockfile.py:blocking_acquire`
- 这说明第一次进入 `wait_gpu_stage()` 时，`python/sglang/jit_kernel/eplb_async_signal.py` 还没有完成 JIT 编译。
- 当这次首次加载恰好发生在 cudagraph capture 路径里时，线程会阻塞在 JIT 编译锁，而不是继续进入实际的 capture。

## 问题根因

- 当前 `eplb_wait_gpu_stage` / `eplb_set_cpu_stage` 改成了 lazy JIT kernel。
- 如果首次调用发生在 `DeepEPMoE.forward()` 的 capture 过程中，就会在 `load_jit(...)` 内部触发编译和锁竞争。
- cudagraph capture 路径不适合临时做 JIT 编译这种重操作，因此表现为 capture hang。

## 改动内容

1. 在 `python/sglang/jit_kernel/eplb_async_signal.py` 中新增：
   - `warmup_eplb_async_signal_module()`
2. 在 `ExpertLocationUpdater.prepare_async_layers()` 中新增 JIT 预热：
   - 在所有 async layer state 初始化之前，先调用 `warmup_eplb_async_signal_module()`
   - 这样首次真正进入 `wait_gpu_stage()` / `set_cpu_stage()` 时，不再触发 `load_jit(...)`
3. 保留原有 fallback：
   - 如果 JIT 预热失败，仍然允许后续走 Python fallback，但会打 warning

## 预期效果

- 避免首次 cudagraph capture 时卡在 JIT 编译锁。
- 将 JIT 编译时机前移到 model 初始化 / async layer 准备阶段。
- 不改变当前 `main -> aux -> main` 的同步语义。
