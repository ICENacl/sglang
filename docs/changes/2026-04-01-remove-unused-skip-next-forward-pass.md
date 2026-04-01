# 删除无效的 skip_next_forward_pass 残留代码

这次修改清理了 `expert_distribution` 中已经没有调用方的 `skip_next_forward_pass` 残留逻辑。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中的：
  - `ExpertDistributionRecorder.skip_next_forward_pass()`
  - `_ExpertDistributionRecorderReal.skip_next_forward_pass()`
  - `_skip_next_forward_pass` 状态位
  - `with_forward_pass()` 中围绕该状态位的条件分支

## 变更原因

- 当前工作区里已经没有任何地方再调用 `skip_next_forward_pass`。
- 继续保留这套状态位和条件分支，只会增加阅读成本，也会让人误以为 async rebalance 还依赖“跳过下一次 forward 统计窗口”的旧逻辑。

## 结果

- `with_forward_pass()` 逻辑更直接。
- 删除了没有调用方的无效接口和状态分支。
