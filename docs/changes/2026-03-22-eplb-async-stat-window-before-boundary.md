# eplb async 边界 iteration 统计调整记录

## 背景

当前 `eplb async` 会在第 `N` 个 iteration 结束后触发 `dump_record()` 和 `rebalance()`。

这会带来两个问题：

1. 第 `N` 个 iteration 的 expert 分布统计会被算进本轮 rebalance 窗口。
2. 即使本轮 rebalance 实际上只需要前面 iteration 的统计，CPU 侧在消费这些统计时，`_copy_to_pinned_cpu_if_needed()` 仍然可能和第 `N` 个 iteration 的执行流发生不必要的同步。

以 `--eplb-rebalance-num-iterations 300` 为例，原来的行为更接近：

- 第 1 到 300 个 iteration 都参与统计。
- 第 300 个 iteration 结束后才做 rebalance。

这不符合 async 路径希望的窗口语义。

## 本次改动

这次不改 rebalance 的触发时机，只调整边界 iteration 的统计行为：

1. 在第 `N-1` 个 iteration 结束时，只记录当前统计流的完成 event，并标记下一个 forward 跳过统计。
2. 第 `N` 个 iteration 开始时，先等待这个 event，再真正 `clone()` 出 snapshot，然后执行 `dump_record()` 和 EPLB metadata 计算。
3. 第 `N` 个 iteration 正常执行 forward，但它自己的 expert 分布不会进入当前 rebalance 窗口。
4. 第 `N` 个 iteration 结束后，再提交这次 rebalance 产出的 metadata 更新计划。
5. 因为 async 计划只会在下一次 `forward start` 激活，所以 `forward N+1` 才会开始对应的 eplb 权重交换。

因此在 `--eplb-rebalance-num-iterations 300` 下，新的行为是：

- 第 1 到 299 个 iteration 的统计被累计到窗口里。
- 第 299 个 iteration 结束时，前 299 个 iteration 的统计对应完成 event 已被记录，但 snapshot 还未真正物化。
- 第 300 个 iteration 开始时，rebalance 会直接消费这份快照并计算新的 metadata。
- 第 300 个 iteration 正常执行，但它的 expert 分布不会写进当前 rebalance 窗口。
- 即使第 300 个 iteration 里有 async metadata publish 对 live 统计张量做 layer reset，也不会再污染这份冻结快照。
- 第 300 个 iteration 结束后，才提交这次 rebalance 产出的 metadata 更新计划。
- 第 301 个 iteration 开始时，async 计划被激活并开始权重交换。

## 代码改动

- `python/sglang/srt/eplb/eplb_manager.py`
  - async 时序改成三段式：
    - `forward N-1 end`：记录统计完成 event，并标记 `forward N` 跳过统计
    - `forward N start`：消费冻结快照并计算 rebalance metadata
    - `forward N end`：提交 metadata 更新计划
  - 边界判断改成直接使用 `forward_pass_id`，避免独立计数器造成 off-by-one。
- `python/sglang/srt/eplb/expert_distribution.py`
  - 新增 `skip_next_forward_pass()`，用于：
    - 在 async scheduler 下记录当前 `_async_logical_count` 统计流的完成 event
    - 丢弃 rebalance iteration 的统计
  - 新增 `materialize_async_snapshot()`：
    - 在 `forward N start` 先做 stream 级 `wait_event`
    - 然后再真正 `clone()` 出稳定 snapshot
    - 不使用 host 阻塞式 `synchronize()`
  - `dump_record(output_mode="object")` 在 async 下优先返回这份冻结快照，而不是直接读取 live `_async_logical_count`。
- `python/sglang/srt/eplb/expert_location.py`
  - 删除 `_copy_to_pinned_cpu_if_needed()`。
  - CPU EPLB 算法和 metadata CPU mirror 都改成直接 `to("cpu")`，需要 pinned memory 时再单独 `pin_memory()`，不再显式做 stream 同步。
  - `ExpertLocationMetadata._init_raw()` 里的 CPU mirror 改成 pinned memory 分配后使用 `copy_(..., non_blocking=True)`，用异步 H2D 对应的 host mirror 形式保存 metadata。
- `python/sglang/srt/model_executor/model_runner.py`
  - 在进入 recorder 前先让 `EPLBManager` 标记当前 forward 是否需要跳过统计。

## 结果

这样改完后，rebalance 的整体执行顺序保持不变，但：

- 统计窗口不再包含边界 iteration 自身
- rebalance 使用的是在 `forward N start` 物化、但只包含前 `N-1` 个 iteration 的稳定快照
- snapshot 的真正物化发生在 `forward N start`，因此不会在 `forward N-1 end` 提前读取仍在 device 上更新的 live 统计
- `forward N` 内的 async layer reset 不会再改坏本轮 rebalance 要消费的统计
- async scheduler 下，即使 `on_forward_pass_end()` 返回时 device 计算尚未真正结束，也不会过早读取 live `_async_logical_count`
- `forward N start` 不再因为 snapshot wait 使用 host 阻塞式 `synchronize()` 而卡住 scheduler
- metadata 计算前移到 `forward N start`
- metadata 更新提交后移到 `forward N end`
- `forward N+1` 开始时才会激活这次 rebalance 对应的 async 权重交换
- metadata 计算仍然保留 `7dc5b90a1` 引入的异步 CPU mirror 路径
