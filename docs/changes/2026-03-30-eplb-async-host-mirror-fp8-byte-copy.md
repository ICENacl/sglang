# EPLB async host mirror 修复 CPU float8 index_copy

这次修改只修复 `EPLB async host mirror` 在 CPU 侧处理 `float8` expert 权重时触发的算子不支持问题，不改变 mirror 的布局、跨节点补齐流程和后续 `host mirror -> live weight` 的 H2D 语义。

## 改动内容

- 在 `python/sglang/srt/eplb/eplb_async_host_mirror.py` 中新增 `_index_copy_host_tensor()`。
- 对普通 dtype，继续沿用原来的 `tensor.index_copy_()`。
- 对 `torch.float8_*` dtype，不再调用 CPU `index_copy_` 的 float8 路径，而是把源/目标张量重解释为按 expert 行组织的 `uint8` 视图，再执行字节级 `index_copy_`。
- 本地 shard 填充和跨节点接收后的回填两条路径都切换到这个辅助函数，避免 CPU `index_copy_cpu` 命中 `float8` 未实现分支。

## 变更原因

- `record.tensor` 和 `packed/cpu_transfer` 都位于 CPU。
- 当 expert 权重 dtype 是 `float8` 时，原实现会在 CPU 上直接执行 `index_copy_`，触发 `index_copy_cpu not implemented for Float8`。
- host mirror 这里需要的只是保留 expert 权重的原始字节内容，并不需要在 CPU 上对 `float8` 做数值计算，因此按字节拷贝即可满足语义。

## 结果

- `float8` expert 权重现在可以正常写入 `/dev/shm` host mirror。
- 非 `float8` dtype 的行为保持不变。
