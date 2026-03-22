# EPLB dispatch map 下沉到 C++ 多线程

本次修改将 `compute_logical_to_rank_dispatch_physical_map()` 的主实现下沉到 C++ 扩展，并使用 `at::parallel_for` 并行计算。

涉及文件：

- `python/sglang/srt/eplb/cpp_expert_location.py`
- `python/sglang/jit_kernel/csrc/eplb_expert_location.cpp`
- `python/sglang/srt/eplb/expert_location.py`

## 改动点

- Python 主路径改为优先调用 `compute_logical_to_rank_dispatch_physical_map_cpp(...)`
- C++ 扩展只构造当前 `ep_rank` 需要的 `[num_layers, num_logical_experts]` 结果
- 使用 `at::parallel_for` 按 `(layer_id, logical_expert_id)` 扁平并行
- Python 实现保留为 fallback，扩展加载或执行失败时自动回退

## 与 TRT-LLM 的对齐点

- 继续沿用“placement / dispatch 决策放在 CPU 侧构造”的思路
- 不把这类不规则 placement 逻辑硬塞进 Triton kernel
- 最终只把构造好的结果一次性搬到 device

## 注意事项

- C++ 路径使用 per-item 派生 seed 来保证并行下的确定性
- 它不保证与旧 Python 串行 `random.Random(seed)` 的逐项随机序列完全一致
- 语义保持一致：same-GPU 优先、same-node 次优、缺失节点走公平补洞
