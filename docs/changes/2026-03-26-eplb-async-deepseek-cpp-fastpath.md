# EPLB async 路径优先使用 DeepSeek C++ rebalancer

本次修改没有改变 DeepSeek EPLB 原本的分配逻辑，只是把原来 `deepseek.py` 中那套 CPU 版重平衡流程按相同语义搬到了 C++。

改动内容：

- 新增 JIT C++ 扩展 `python/sglang/jit_kernel/csrc/eplb_deepseek.cpp`
  - 复刻 `balanced_packing`
  - 复刻 `replicate_experts`
  - 复刻 `rebalance_experts_hierarchical`
  - 复刻 `rebalance_experts`
- 新增 Python 包装 `python/sglang/srt/eplb/cpp_deepseek.py`
- 在 `python/sglang/srt/eplb/expert_location.py` 中调整 `init_by_eplb()`
  - 当 `enable_eplb_async=True`
  - 且算法为 `deepseek` 或 `deepseek_hierarchical`
  - 优先调用新的 C++ rebalancer
  - 如果 C++ 扩展加载或执行失败，则记录告警并回退到原 Python 实现

这样处理后：

- 同步 EPLB 路径保持原样
- async 路径会优先避开 Python 热循环
- 算法输出保持和原 DeepSeek CPU 逻辑一致
