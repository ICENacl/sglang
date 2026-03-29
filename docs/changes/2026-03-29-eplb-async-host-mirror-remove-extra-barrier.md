# EPLB async host mirror 去掉多余 barrier

这次修改只去掉 `EPLB async host mirror` 构建阶段一条不影响正确性的全局 barrier，不改变 mirror 数据内容、跨节点补齐流程和完成性校验语义。

## 变更

- 删除 `python/sglang/srt/eplb/eplb_async_host_mirror.py` 里 `_create_records()` 末尾的第二条 `_barrier_all_ranks()`。
- 保留其余 barrier：
  - owner 创建完 `/dev/shm` 记录后，非 owner 再 attach 的同步点；
  - 本地 shard 填充完成后，再开始跨节点补齐的同步点；
  - 跨节点补齐结束后，再执行完整性校验和 attach 的同步点；
  - leader 之间按 layer 顺序推进远端传输的同步点。

## 原因

- `_create_records()` 的第一条 barrier 已经保证 owner 先创建好共享内存对象，非 owner 才开始打开并 attach。
- 非 owner attach 完成后，owner 不需要再等待所有 rank 都 attach 完，后续本地 shard 写入和远端补齐仍然会被后面的 barrier 正确保护。
- 去掉这条 barrier 可以减少 build 阶段一次纯等待，不改变 host mirror 的最终可见数据。
