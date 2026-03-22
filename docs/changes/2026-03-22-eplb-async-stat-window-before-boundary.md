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

1. 当即将进入第 `N` 个 iteration 时，提前把这一次 forward 标记成“跳过 expert 分布统计”。
2. 第 `N` 个 iteration 正常执行 forward。
3. 第 `N` 个 iteration 结束后，仍然沿用原来的路径触发 `dump_record()` 和 `rebalance()`。

因此在 `--eplb-rebalance-num-iterations 300` 下，新的行为是：

- 第 1 到 299 个 iteration 的统计被累计到窗口里。
- 第 300 个 iteration 正常执行，但它的 expert 分布不会写进统计窗口。
- 第 300 个 iteration 结束后触发 rebalance，此时消费到的仍然只是前 299 个 iteration 的统计。

## 代码改动

- `python/sglang/srt/eplb/eplb_manager.py`
  - 新增 async 路径的 `on_forward_pass_start()`。
  - 在边界 iteration 开始前仅标记“跳过本次统计”。
- `python/sglang/srt/eplb/expert_distribution.py`
  - 新增 `skip_next_forward_pass()`，用于丢弃 rebalance iteration 的统计。
- `python/sglang/srt/eplb/expert_location.py`
  - 删除 `_copy_to_pinned_cpu_if_needed()`。
  - CPU EPLB 算法和 metadata CPU mirror 都改成直接 `to("cpu")`，需要 pinned memory 时再单独 `pin_memory()`，不再显式做 stream 同步。
  - `ExpertLocationMetadata._init_raw()` 里的 CPU mirror 改成 pinned memory 分配后使用 `copy_(..., non_blocking=True)`，用异步 H2D 对应的 host mirror 形式保存 metadata。
- `python/sglang/srt/model_executor/model_runner.py`
  - 在进入 recorder 前先让 `EPLBManager` 标记当前 forward 是否需要跳过统计。

## 结果

这样改完后，rebalance 的整体执行顺序保持不变，但统计窗口不再包含边界 iteration 自身，async 统计到 CPU 的这条路径也不再依赖 `_copy_to_pinned_cpu_if_needed()` 里的显式 stream 同步。
