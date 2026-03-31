# EPLB Async logical count 三阶段时序调整

本次调整把 `eplb async` 的 manager 时序拆成了 3 个 iter，目标是避免在 host scheduler 所在的 `forward end` 路径上，为了获取 `logical_count` 而同步等待当前 iter 的 graph 完成。

## 调整前

原来的 async 路径是两阶段：

1. 在 rebalance 边界 iter 的 `forward end` 中：
   - `dump_record()`
   - 生成 `logical_count`
   - 拷到 CPU
   - `init_by_eplb(...)`
2. 下一次 `forward end`：
   - `apply/submit plan`

问题在于：

- 当前 iter 的 graph 还可能在设备上执行；
- `logical_count` 的 CPU 拷贝如果在这时同步等待，会直接阻塞 host scheduler；
- 如果不显式等当前计算流，又容易把 `logical_count` 的读取时序写得不够清楚。

## 调整后

现在改成三阶段：

1. rebalance 边界 iter：
   - `dump_record()`
   - 生成 `logical_count`
   - 保留 GPU 上的 `logical_count`
   - async step 对应的 `physical_to_logical_map` 也保留在 GPU buffer 中
   - 在当前计算流上记录 ready event
   - 不在这个阶段做 CPU 拷贝，也不在 host 上等待

2. 下一次 `forward end`：
   - 只有当 `logical_count` 的 ready event 已完成时，才继续
   - manager 直接把 GPU `logical_count` 传给 `init_by_eplb(...)`
   - 明确在 `prepare_stream` 上执行 `init_by_eplb(...)`
   - 由 `init_by_eplb(...)` 自己按算法需求决定是否转成 pinned CPU tensor
   - 生成待应用的 metadata

3. 再下一次 `forward end`：
   - `apply/submit plan`

## 新语义

- `logical_count` 仍然来自 rebalance 边界那一轮的 recorder 窗口，不额外前移统计窗口。
- 但 `logical_count` 的 CPU 消费和 `init_by_eplb(...)` 被延后到下一轮。
- `apply/submit` 再延后一轮，保证时序清楚：
  - iter N: fetch logical count
  - iter N+1: init metadata
  - iter N+2: apply

## 为什么这样改

这样改以后：

- `logical_count` 的生产仍然排在当前计算流上；
- rebalance 边界 iter 不再做额外 D2H copy；
- `dump_record()` 生成 `logical_count` 时不再需要把 step mapping 从 CPU 搬回 GPU，因此去掉了这段 H2D；
- manager 不再判断 `logical_count` 是否需要 CPU；
- 只有在 `deepseek` / `deepseek_hierarchical` async C++ 路径需要 CPU `logical_count` 时，才由 `init_by_eplb(...)` 在下一阶段转成 pinned CPU tensor；
- host 侧不再在 rebalance 边界 iter 上因为 D2H ready event 被卡住；
- `init_by_eplb(...)` 和 `apply` 分别在后续 iter 推进，语义更清楚。

## 代码位置

- manager 时序调整：
  - [eplb_manager.py](/config/workspace/sglang/python/sglang/srt/eplb/eplb_manager.py)
- 对应测试：
  - [test_qwen3_moe_eplb.py](/config/workspace/sglang/test/srt/test_qwen3_moe_eplb.py)
