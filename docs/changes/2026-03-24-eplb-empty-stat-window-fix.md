# EPLB 空统计窗口除零修复

## 问题

`_DequeCollection.mean()` 在统计窗口被 `clear()` 清空后，仍然会对空 `deque` 执行 `sum(d) / len(d)`。

这会触发 `ZeroDivisionError`，并在读取 balancedness 历史均值时中断流程。

## 本次改动

- `python/sglang/srt/eplb/expert_distribution.py`
  - 在 `mean()` 中跳过空窗口，只返回已有样本的统计结果

- `test/srt/test_expert_distribution_unittest.py`
  - 新增单测，覆盖 `clear()` 后再次读取均值的场景

## 结果

- 空统计窗口不会再触发除零
- 有样本时的均值计算逻辑保持不变
