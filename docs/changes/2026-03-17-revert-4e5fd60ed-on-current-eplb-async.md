# 基于当前工作区回退 4e5fd60ed 的说明

## 背景

- `4e5fd60ed0808cc03faee27f7ed7784be744e837` 引入了两类东西：
  - 基础设施：`PublishedExpertLocationMetadata` 以及围绕它的按 layer publish 包装
  - 语义：把 async metadata publish 放到 `next forward start`
- 当前工作区已经沿着后续方案继续演进：
  - async 统计改成按事件直接累计 logical count
  - H2D 完成后立即 publish layer metadata
  - layer commit 后立刻清零该 layer 的统计窗口
- 用户确认 `4e5fd60ed` 存在精度问题，因此这里需要在当前工作区基础上撤掉这笔提交本身引入的实现形态。

## 本次回退

- 删除 `PublishedExpertLocationMetadata` 包装，恢复全局 `expert_location_metadata` 直接使用 `ExpertLocationMetadata`。
- `model_runner` 初始化时不再为 async 额外包装 metadata。
- `expert_location_updater` 的 layer publish 改为直接调用：
  - `current_metadata.update(new_metadata, update_layer_ids=[layer_id])`
- 同时保留当前工作区后续新增的逻辑：
  - H2D 完成后立即 publish layer metadata
  - layer 级 async logical 统计清零
  - `topk.py` 中统一收口的 logical 统计逻辑
- 一并删除 `4e5fd60ed` 自带的变更说明文档，避免文档继续指向已经回退的“next forward start publish”语义。

## 结果

- 当前 async 方案不再依赖 `4e5fd60ed` 引入的 wrapper 结构。
- 但仍保留后续修复所需的核心能力：
  - copy completion 立即生效
  - layer 粒度统计窗口切换
