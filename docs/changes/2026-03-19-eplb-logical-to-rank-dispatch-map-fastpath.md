# compute_logical_to_rank_dispatch_physical_map 快路径优化

本次修改优化了 `python/sglang/srt/eplb/expert_location.py` 中
`compute_logical_to_rank_dispatch_physical_map()` 的构造开销。

## 原问题

原实现存在三个明显低效点：

1. 最终只返回当前 `ep_rank` 的 `[num_layers, num_logical_experts]` 结果，
   但中间会先构造完整的 `(ep_size, num_layers, num_logical_experts)` 张量。
2. 内层循环使用 `torch.sum(...).item()` 和张量布尔索引，Python 小循环里频繁触发张量路径。
3. 输入如果在 device 上，还会通过 `_logical_to_all_physical_raw(...).tolist()` 产生额外等待。

## 修改

- 先把输入统一收敛到 CPU cache
- 只构造当前 rank 需要的 dispatch map，输出形状直接为 `[num_layers, num_logical_experts]`
- 内层逻辑改成“当前 rank 直接求值”：
  - 不再显式遍历所有 `ep_rank`
  - 通过候选 expert 的 GPU/node 分布，直接判断当前 rank 是否命中 same-GPU / same-node
  - 仅在确实存在缺失节点时才调用 `fair_choices`
  - 同时保持与旧实现一致的随机数消费顺序
- 最后再一次性把结果搬回目标 device

## 预期效果

- 去掉 `(ep_size, ...)` 级别的中间张量分配
- 去掉内层对所有 `ep_rank` 的显式扫描
- 去掉 `torch.sum(...).item()` 和布尔张量写入
- 减少 dispatch map 构造阶段的 host/device 同步等待
