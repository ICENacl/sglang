# 删除无效的 include_in_rebalance_window 参数链

这次修改清理了 `expert_distribution` 中已经退化为恒定值的 `include_in_rebalance_window` 参数链。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中以下位置对 `include_in_rebalance_window` 的传递和判断：
  - `_on_forward_pass_end()`
  - `_Accumulator.append()` 及其子类实现
  - `_UtilizationRateAccumulatorMixin._append_utilization_rate()`
- 删掉 async 分支里基于该参数的无效条件判断。

## 变更原因

- 当前 `with_forward_pass()` 已经固定传入 `include_in_rebalance_window=True`。
- 也就是说，这个参数已经不再承载实际分支语义，只剩无效透传。
- 保留这条参数链会增加理解成本，也会让人误以为当前 recorder 还存在“有条件跳过 rebalance window”的逻辑。

## 结果

- recorder 内部参数传递更直接。
- 删除了恒定参数带来的无效代码路径。
