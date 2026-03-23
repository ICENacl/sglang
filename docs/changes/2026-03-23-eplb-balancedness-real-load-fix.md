# eplb balancedness 真实负载修复

## 问题

`current_pass_balancedness` 原来直接复用 recorder 主路径里的 `global_physical_count`。

这会导致它统计到的并不一定是“本次 forward 的真实 physical load”：

- `moe_a2a_backend == none` 时，真实负载来自 physical `topk_ids`
- `deepep normal` 时，真实负载来自 dispatcher 的 `local_physical_count_of_layer`
- `deepep low_latency` 时，真实负载来自 low-latency dispatch 的 local physical count

把这些模式都折叠到同一条 recorder 输入上，会让 `current_pass_balancedness` 在开启 EPLB 后统计口径失真。这个问题不只发生在 `--enable-eplb-async` 下，sync 模式也会受影响。

## 本次改动

- `python/sglang/srt/eplb/expert_distribution.py`
  - 新增 `metric_global_physical_count`，专门给 `current_pass_balancedness` 使用。
  - 按实际运行模式分别记录本次 forward 的真实 physical load：
    - `topk_physical`
    - `dispatch_normal`
    - `dispatch_low_latency`
  - `_append_utilization_rate()` 优先使用 `metric_global_physical_count`，不再直接依赖 recorder 主路径的 `global_physical_count`。

## 结果

- `current_pass_balancedness` 改为反映真实的 physical load
- 不再依赖被 recorder / rebalance 复用过的统计口径
- sync 和 async 模式都会走同一套真实负载统计逻辑
