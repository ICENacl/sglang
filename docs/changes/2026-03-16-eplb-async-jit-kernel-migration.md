# eplb async signal kernel 改为 jit kernel 记录

## 改动原因

- 当前 `eplb async` 的 signal kernel 规模很小，只服务于 `ExpertLocationUpdater` 的 wait/set 同步点。
- 继续把这类 kernel 放在 `sgl-kernel` 全量扩展里，会增加编译链路和注册维护成本。
- 仓库里已有 `python/sglang/jit_kernel/hicache.py`、`python/sglang/jit_kernel/cuda_wait_value.py` 这类按需 JIT 编译的实现，可以直接复用同样模式。

## 改动内容

1. 新增 JIT kernel 文件
   - `python/sglang/jit_kernel/eplb_async_signal.py`
   - `python/sglang/jit_kernel/csrc/eplb_async_signal.cuh`
2. `ExpertLocationUpdater`
   - `wait_gpu_stage()` 改为调用 JIT wrapper `jit_eplb_wait_gpu_stage(...)`
   - `set_cpu_stage()` 改为调用 JIT wrapper `jit_eplb_set_cpu_stage(...)`
   - 若 JIT 加载失败，保留 Python fallback，自旋/host 写位逻辑仍可工作
3. `sgl-kernel`
   - 删除这次新增的 `eplb_async_commit.cu`
   - 清理 `sgl_kernel_ops.h` 和 `common_extension.cc` 中的 signal op 注册
4. JIT wrapper 轻量化
   - `eplb_async_signal.cuh` 不再依赖 `sgl_kernel/tensor.h`
   - 改为直接使用 `tvm::ffi::TensorView + utils.cuh`
   - 删除 `TensorMatcher/SymbolicSize` 这类重模板校验，只保留最小设备与 dtype 检查

## 影响

- `eplb async` 的 signal kernel 不再依赖 `sgl-kernel` 扩展导出。
- 首次进入该路径时会按需触发 JIT 编译。
- 当前环境仍未做真实 CUDA 运行验证，只完成了静态修改。
