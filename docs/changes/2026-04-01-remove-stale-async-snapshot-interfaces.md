# 删除失效的 async snapshot 残留代码

这次修改清理了 EPLB async 主流程已经不再使用的 snapshot 残留代码。

## 改动内容

- 删除 [expert_distribution.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py) 中不再被实际主流程使用的内容：
  - `AsyncRebalanceSnapshot`
  - `materialize_async_snapshot()`
  - `prepare_async_rebalance_snapshot()`
  - `detach_async_rebalance_global_physical_count()`
  - `detach_async_rebalance_physical_to_logical_map()`
- 删除 [eplb_manager.py](/config/workspace/sglang/python/sglang/srt/eplb/eplb_manager.py) 中对空实现 `materialize_async_snapshot()` 的无效调用。

## 变更原因

- 当前 async rebalance 已经改为直接通过 `dump_record(output_mode="object")` 获取 `logical_count`，再在 `prepare_stream` 上准备 metadata。
- 旧的 snapshot / detach buffer 路径已经不再参与主流程。
- 保留这些接口只会增加阅读和维护成本。

## 结果

- async rebalance 代码路径更直接。
- 删除了无效接口和空调用，减少歧义。
