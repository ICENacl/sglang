# EPLB Sync 统计语义回退到 v0.5.9

这次修改只修复 `eplb async` 引入的公共统计逻辑对 `eplb sync` 的污染，不回退 async 的 prepare/apply、host mirror、metadata 发布和 layer reset 语义。

具体行为：

- `eplb sync` 的 `current_pass_balancedness` 重新按 `v0.5.9` 使用 `global_physical_count` 计算。
- `eplb sync` 的 EPLB 输入重新按 `v0.5.9` 走全局窗口统计和全局聚合。
- `deepseek.py` 整体回退到 `v0.5.9` 的算法实现。
- `eplb async` 继续保留当前的 boundary skip、runtime reset 和 group 内聚合语义。
- `topk.py` 不在这次修复范围内。
