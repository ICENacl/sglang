# eplb async 下 expert distribution metrics 统计修正

## 问题

当前 `eplb async` 为了让 rebalance 窗口跳过边界 iteration，会在 `skip_next_forward_pass()` 后整段跳过 recorder。

这会带来一个副作用：

- 该 iteration 的 rebalance 统计确实被跳过了。
- 但 `--enable-expert-distribution-metrics` 依赖的 gatherer 和 metrics 输出也一起被跳过了。

结果就是边界 iteration 的 balancedness 统计会缺失，导致 metrics 不准确。

## 本次改动

- `python/sglang/srt/eplb/expert_distribution.py`
  - `with_forward_pass()` 不再因为 `skip_next_forward_pass()` 而整段跳过 recorder。
  - 改成继续执行 gatherer 和 forward 结束时的 metrics 计算，但通过 `include_in_rebalance_window` 标记控制是否写入 rebalance 窗口。
  - async logical count 的更新在边界 iteration 仍然会被跳过，因此不会污染 rebalance snapshot。
  - `_StatAccumulator` 和 `_DetailAccumulator` 只会在 `include_in_rebalance_window=True` 时写入 recorder 窗口数据。

## 结果

- rebalance 窗口语义保持不变，边界 iteration 仍然不会进入当前 EPLB 统计窗口。
- `--enable-expert-distribution-metrics` 的 gatherer 和 metrics 输出不再被边界 iteration 一起跳过。
- async EPLB 和 expert balancedness metrics 可以同时保持正确。
