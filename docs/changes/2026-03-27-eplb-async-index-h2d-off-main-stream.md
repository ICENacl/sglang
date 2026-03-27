# EPLB Async index H2D 不再挂在主计算 stream

`eplb async` 在把 buffered `physical_to_logical_map` 从 CPU 拉回 GPU 时，
`index = index.to(device=device, dtype=torch.int64)` 会直接挂在当前 stream 上。

这会带来两个问题：

- H2D copy 会被主计算 stream 上已有工作串行阻塞。
- 如果 CPU buffer 不是 pinned memory，这次 copy 也无法真正异步。

这次修改做了两件事：

- async 路径的 `physical_to_logical_map` CPU buffer 改成 pinned memory。
- `_convert_global_physical_count_to_logical_count()` 在 CPU -> CUDA 的 index 转换场景下，
  使用单独的 CUDA stream 完成 H2D 和 `scatter_add_`，最后再把依赖回接到当前 stream。

这样不会改变逻辑结果，但可以避免这次 index 转换继续占住主计算 stream。
