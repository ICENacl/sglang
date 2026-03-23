# current_pass_balancedness 按 forward 清零修复

## 问题

前一版为 `current_pass_balancedness` 增加了独立的 `metric_global_physical_count`，但这份计数只在 recorder 整体 `_reset()` 时清零。

这会导致它在多个 forward 之间持续累加，而不是只统计“当前这一拍”的 physical load。表现出来就是：

- `current_pass_balancedness` 不再反映单次 forward
- 数值会随着请求推进缓慢增长或缓慢变化

## 本次改动

- `python/sglang/srt/eplb/expert_distribution.py`
  - 在 `_on_forward_pass_start()` 里先把 `metric_global_physical_count` 清零
  - 让这份 metrics 计数严格按单个 forward 统计

## 结果

- `current_pass_balancedness` 回到“current pass”的语义
- 不再跨多个 forward 累加
