# EPLB Async 增加 dummy H2D 调试模式

这次修改增加了一个仅用于调试的环境变量：

- `SGLANG_EPLB_ASYNC_DUMMY_H2D=1`

开启后：

- 跳过 `EPLB async host mirror` 的真实 build。
- 不创建 shm，也不做本地/跨节点 mirror 填充。
- 每次 async H2D 交换权重时，都返回同形状的 fake CPU pinned tensor。
- 当前 fake 数据固定为全 0。

默认关闭时，原有真实 host mirror 路径保持不变。

主要用途：

- 只验证 async control plane / signal / H2D 调度时序。
- 排除 host mirror build 和真实 expert 数据内容对问题定位的影响。
