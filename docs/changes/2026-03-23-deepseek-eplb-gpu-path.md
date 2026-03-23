# deepseek eplb GPU 路径调整

## 背景

当前 `deepseek` EPLB 算法会被归到 CPU 路径，并在实现里直接把输入 `weight` 转成 CPU tensor。

这会让非 hierarchical 的 `deepseek` 在 metadata 计算阶段多出一次不必要的 `GPU -> CPU` 迁移。

## 本次改动

- `python/sglang/srt/eplb/eplb_algorithms/deepseek.py`
  - `deepseek` 的实现改为直接复用 `deepseek_vec` 的张量路径。
  - 输入 `weight` 会先扩成单步的 `tokens_per_expert`，然后调用 `deepseek_vec.rebalance_experts(...)`。
  - 输出仍保持 `phy2log / log2phy / logcnt` 这组接口。
- `python/sglang/srt/eplb/eplb_algorithms/__init__.py`
  - `algorithm_runs_on_cpu()` 不再把 `EplbAlgorithm.deepseek` 归到 CPU 路径。

## 范围

- 这次只把非 hierarchical 的 `deepseek` 改成 GPU 张量路径。
- `deepseek_hierarchical` 仍然保留在 CPU 路径，因为它当前仍依赖 group packing 的 CPU 实现。
