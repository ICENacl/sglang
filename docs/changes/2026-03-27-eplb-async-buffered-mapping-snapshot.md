# EPLB Async 按 Forward 冻结 Mapping 再做 Rebalance 转换

## 背景

`enable_eplb_async=1` 且采用 post-launch prepare 时，rebalance 使用的是一个窗口内累计的 `global_physical_count`。

之前这段统计在 prepare 阶段只会读取“一份当前的 `physical_to_logical_map`”去解释整段窗口里的 physical count。第一轮 async EPLB 发布部分 layer metadata 后，后续窗口里的旧统计会被新 metadata 重新解释，导致第二次及以后 rebalance 容易出现 `copy_pairs=0`，表现为没有 H2D 权重交换。

## 修改

- 在 recorder 中为 async rebalance 额外缓存每个 forward 的 CPU `physical_to_logical_map`
- `with_current_layer()` 进入 layer 前，先把该 layer 当前生效的 CPU mapping 冻结到本次 forward 的快照里
- post-launch prepare 不再直接把 buffered physical count 转成 logical count
- prepare 阶段先 detach：
  - `global_physical_count`
  - 每个 forward 对应的 `physical_to_logical_map`
- 后台线程再按 forward 使用各自冻结的 mapping 做 `scatter_add`
- async `dump_record()` 也改为使用 buffered mapping，而不是直接读取当前 metadata

## 结果

- 保留 `089dceafadef833bdccc23c9b88d55482a202449` 引入的 post-launch prepare 语义，不重新阻塞 host scheduler
- 不在 graph/cudagraph 内新增统计写路径
- 第二次及以后 async EPLB 会基于窗口内真实生效过的 mapping 计算 logical count，避免因为 metadata 解释错位把 `copy_pairs` 错算成 0
