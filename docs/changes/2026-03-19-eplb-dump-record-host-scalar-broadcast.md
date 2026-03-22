# EPLB dump_record 去掉平均利用率的 CUDA 中转

本次修改优化了 `python/sglang/srt/eplb/expert_distribution.py` 中 `dump_record()` 的 host 标量广播路径。

## 背景

此前 `average_utilization_rate_over_window` 的计算虽然本来就在 CPU 上完成，但实现会：

1. 先把该标量包装成 CUDA tensor
2. 在 device group 上做 `broadcast`
3. 再通过 `.item()` 拉回 CPU

这会在 `dump_record()` 边界引入额外的 H2D / D2H 和同步开销。

## 修改

- 保留 `logical_count` 在 `moe ep device group` 上做 device `all_reduce`
- 将 `average_utilization_rate_over_window` 改为直接在 `moe ep cpu group` 上做 `broadcast_object_list`
- 不再为这个 host 标量构造 CUDA tensor，也不再调用 `.item()`

## 与 TRT-LLM 的对齐点

TRT-LLM 的负载统计消费边界偏向 CPU 侧，`expertLoadFactor` 也是 host-visible/pinned 内存。这里没有完全复制 TRT-LLM 的统计结构，但把 `dump_record()` 中本来就属于 CPU 的标量通信改成了纯 host 路径，避免了不必要的设备往返。
