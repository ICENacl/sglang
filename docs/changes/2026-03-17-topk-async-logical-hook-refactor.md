# 收束 topk.py 中的 async logical 统计改动

## 背景

- 之前为了保证 async recorder 统计到的是 remap 前的 logical expert id，`topk.py` 在多个函数里分别插入了 `on_select_experts_logical(...)`。
- 这种改法虽然功能上可行，但改动点过散：
  - `fused_topk`
  - `grouped_topk`
  - `biased_grouped_topk`
  - `forward_npu`
- 后续如果再调整 topk 路径，很容易漏掉某个分支。

## 本次调整

- 新增统一收口 helper：`_finalize_topk_ids(...)`
- 这个 helper 负责三件事：
  - 在 remap 前记录 logical `topk_ids`
  - 按 `expert_location_dispatch_info` 做 logical -> physical remap
  - 处理 padded token 区域的 `-1`
- 普通 `select_experts()` 路径现在统一在函数尾部调用 `_finalize_topk_ids(...)`。
- `TopK.forward_npu()` 因为绕过了 `select_experts()`，保留单独调用 `_finalize_topk_ids(...)`。
- 低层 `fused/grouped/biased` 实现尽量只负责计算原始 `logical_topk_ids`，不再各自夹带 recorder 逻辑。

## 结果

- `topk.py` 中 async logical 统计的核心入口从“多点散落”收束为“少量公共后处理点”。
- 仍然保持关键语义不变：
  - recorder 看到的是 remap 前的 logical id
  - dispatcher / compute 继续使用 remap 后的 physical id
