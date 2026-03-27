# EPLB Async 统计来源不再依赖 metrics 开关

## 背景

之前 async EPLB 在累计 rebalance 窗口时，默认使用 `global_physical_count`。

在 `deepep normal` 场景下，这份默认统计来自 `select_experts`，而不是实际 dispatch 结果。只有在打开 `--enable-expert-distribution-metrics` 后，rebalance 才会改用 `metric_global_physical_count`，这会让第二次及以后 EPLB 的 `copy_pairs` 和 H2D 行为依赖 metrics 开关。

## 修改

- 为 async rebalance 单独维护 `async_global_physical_count`
- 这份统计始终按当前运行后端选择来源：
  - `moe_a2a_backend=none`：使用 topk physical expert
  - `deepep normal`：使用 dispatch 的 `local_physical_count`
  - `deepep low_latency`：使用 low-latency dispatch 的 `local_physical_count`
- `_StatAccumulator.append()` 在 async 下固定使用 `async_global_physical_count`
- `enable_expert_distribution_metrics` 只继续影响观测和指标，不再改变 rebalance 输入

## 结果

- 第二次及以后 async EPLB 是否发生 H2D，不再取决于 metrics 是否开启
- async rebalance 使用的始终是实际运行路径对应的 physical-count 统计
