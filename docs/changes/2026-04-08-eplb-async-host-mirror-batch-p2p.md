# EPLB Async host mirror 跨节点补齐改为 batch P2P

这次修改只调整 `EPLB async host mirror` 构建阶段的跨节点补齐方式，仍然保留原来的 device-group P2P 后端，不引入 `gloo`。

## 改动内容

- `python/sglang/srt/eplb/eplb_async_host_mirror.py`
  - 删除 host mirror 跨节点补齐对 `leader_cpu_group` 的依赖。
  - 同一个 `layer + transfer_group` 下的多个 tensor，不再逐个调用 `dist.send/recv`。
  - 改为为这一组 tensor 构造一批 `torch.distributed.P2POp`，再用 `torch.distributed.batch_isend_irecv(...)` 一次性提交。
  - 发送端仍然是：
    - 从 host mirror shm 取出对应 logical experts
    - 拷到 GPU staging buffer
    - 走 device-group P2P 发给目标 node
  - 接收端仍然是：
    - 先收进 GPU staging buffer
    - 再拷回 pinned CPU buffer
    - 最后写回本地 host mirror shm

## 变更原因

- 旧实现虽然走的是 P2P，但同一个 transfer group 内还是按 tensor 串行发多次消息。
- 这会带来额外的通信启动开销，尤其在每层 tensor 数量较多时更明显。
- 对这类固定数量、同一对端、同一时机的消息，更合适的做法是直接用 batch P2P 一次挂出整组操作。

## 结果

- 保留现有 P2P 后端，不切换到 `gloo` 或磁盘加载。
- 同一 transfer group 的多 tensor 交换改为一次 `batch_isend_irecv` 提交，减少逐条消息调度开销。
- host mirror 的最终数据内容和后续 `host mirror -> live weight` 路径保持不变。
