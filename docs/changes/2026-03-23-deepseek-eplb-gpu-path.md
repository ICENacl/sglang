# deepseek eplb GPU 路径调整

## 背景

当前 `deepseek` EPLB 算法会被归到 CPU 路径，并且在入口里直接把 `weight` 转成 CPU tensor。

这会让非 hierarchical 的 `deepseek` 在 metadata 计算阶段发生一次不必要的 `GPU -> CPU` 迁移。

## 本次改动

- `python/sglang/srt/eplb/eplb_algorithms/deepseek.py`
  - 保留 `deepseek` 原有算法，不复用 `deepseek_vec`。
  - 原始 CPU 版 `balanced_packing` 保留为独立实现。
  - 新增 GPU 版 `balanced_packing`，并用 Triton kernel 重写逐步 pack 选择。
  - `replicate_experts` 的 GPU 路径也改为 Triton kernel 驱动的逐步更新。
  - 默认的 `balanced_packing()` 入口改为走 GPU 版本；CPU 版本只由 hierarchical 路径显式调用。
  - `rebalance_experts()` 只在 hierarchical 模式下继续转 CPU；非 hierarchical 模式直接在当前设备上计算。
  - `log2phy` 的第三维改为使用副本数理论上界，去掉 `logcnt.max().item()` 带来的同步拷贝。
- `python/sglang/srt/eplb/eplb_algorithms/__init__.py`
  - `algorithm_runs_on_cpu()` 不再把 `EplbAlgorithm.deepseek` 归到 CPU 路径。

## 范围

- 这次只把非 hierarchical 的 `deepseek` 改成 GPU 张量路径。
- `deepseek_hierarchical` 仍然保留在 CPU 路径，因为它当前仍依赖 hierarchical 的 CPU 调度过程。

## 验证

- 新增 `test/srt/test_eplb_algorithms.py`
  - 验证 `EplbAlgorithm.deepseek` 不再被 `algorithm_runs_on_cpu()` 归类为 CPU-only。
  - 验证 `balanced_packing()` 默认走 GPU 实现。
  - 验证原始 CPU 的 `balanced_packing` 实现仍然保留。
  - 验证 GPU 包装层在非 Triton 条件下仍能回退到原始实现。
  - 验证非 hierarchical 的 `deepseek` 结果与原始 CPU 算法一致。
  - 验证 `log2phy` 在保留有效结果的同时，剩余位置会被 `-1` 正确填充。
