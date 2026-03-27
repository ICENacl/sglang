# EPLB Async CUDA Tensor 统一落到当前 rank 的 current device

这次修改针对 `enable_eplb_async` 相关路径里一类容易在多 rank 下出错的 device 选择问题。

## 问题

async EPLB 的 metadata 构建、统计累计和 canary 校验里，原来有几处 tensor 创建或 `.to(...)` 直接使用了 `server_args.device`。

当多 rank 进程的当前 CUDA device 和这个配置值不一致时，非 rank0 可能会把本 rank 的 tensor 建到错误的 GPU 上，表现为：

- rebalance prepare / apply 期间出现 device mismatch
- 非 rank0 在 async 路径访问了不属于当前 rank 的 CUDA tensor

## 修改

- `ExpertLocationMetadata.init_by_mapping()` / `init_by_eplb()` / `_init_raw()`
  - 优先跟随已有 expert metadata / 输入 tensor 自身所在的 device
- `ExpertDistributionRecorder`
  - async 统计 tensor
  - metric 统计 tensor
  - 平均利用率广播前的临时 tensor
  - 统一跟随当前 expert metadata 的 device
- `ExpertLocationUpdater`
  - canary tensor 改为直接拷到对应 `routed_experts_weights` 的 device

## 结果

- async EPLB 路径不再依赖 `server_args.device` 去决定 CUDA tensor 的落点
- 每个 rank 都跟随本 rank 已生效的 metadata / 权重 device 去分配和使用这些 tensor
