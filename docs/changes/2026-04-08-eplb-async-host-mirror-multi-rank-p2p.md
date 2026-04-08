# EPLB Async host mirror 跨节点补齐改为多 rank 并行 P2P

这次修改继续只针对 `EPLB async host mirror` 的构建阶段，不改运行时更新逻辑。

## 改动内容

- `python/sglang/srt/eplb/eplb_async_host_mirror.py`
  - 跨节点补齐不再只让每个 node 的 `local_rank == 0` 参与。
  - 新增按 logical expert 分配发送 rank 和接收 rank 的计划生成逻辑：
    - 发送端使用当前 node 内真正持有该 expert node-owner 副本的 local rank。
    - 接收端使用稳定的 `logical_expert_id % local_world_size` 分片。
  - 每个 logical expert 只会落到唯一的一对 `(src local rank, dst local rank)`，避免多个 rank 重复写同一个 shm 槽位。
  - 真正的跨节点通信仍然使用 device-group P2P，并继续通过 `batch_isend_irecv` 一次提交同一个 transfer group 的多个 tensor。

## 变更原因

- 之前即使已经改成了 batch P2P，跨节点补齐仍然只有每个 node 的单个 rank 在传。
- 这会把 host mirror 初始化期的跨节点带宽和调度压力集中到一个 rank 上，容易形成单点瓶颈。
- host mirror 本身是 node 内共享 shm，只要目标 expert 的写入责任分配清楚，就可以让多个本地 rank 同时参与补齐。

## 结果

- host mirror 的跨节点补齐从“每个 node 单 rank”变成了“每个 node 多 rank 分片并行”。
- 发送和接收责任都稳定且唯一，不会引入重复写同一个 logical expert 的问题。
- 这一改动和已有的 batch P2P 叠加后，跨节点补齐的并行度会明显高于之前的 leader-only 设计。
