# EPLB async host mirror 按需复用 staging buffer

这次修改只收紧 `EPLB async host mirror` 构建期间的 staging buffer 生命周期，不改变 host mirror 数据内容、跨节点补齐语义和后续 `host mirror -> live weight` 的 H2D 路径。

## 改动内容

- `python/sglang/srt/eplb/eplb_async_host_mirror.py`
  - 去掉为每个 `layer/tensor` 预先创建一份常驻 `gpu_staging_buffer` 和 `cpu_transfer_buffer` 的做法。
  - 改为按 `(dtype, device, tensor.shape[1:])` 复用 GPU staging buffer。
  - 改为按 `(dtype, tensor.shape[1:])` 复用 pinned CPU transfer buffer。
  - buffer 仅在当前容量不足时按需扩容，扩容后继续复用。
  - 记录每个 tensor 对应的原始 device，用于跨节点传输时在正确的 GPU 上申请 staging buffer。

## 变更原因

- 原实现会为每个 tensor record 常驻保留一份 GPU staging buffer。
- host mirror build 本身是串行按 tensor 处理的，这些 buffer 并不会并发使用。
- 因此显存占用会随着 tensor record 数量增长，但这部分增长并不是必须的。

## 结果

- 构建 host mirror 期间的常驻显存从“按 tensor 数量累加”收敛为“按不同 tensor 规格复用”。
- 仍然保留按需扩容能力，不影响跨节点批量 expert 传输。
