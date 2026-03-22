## 背景

部分 MoE 模型把 `routed_experts_weights_of_layer` 保存为 `LazyValue`，而 EPLB 异步路径默认直接把它当作字典使用。

## 问题

这会在访问 `.items()` 或 `.keys()` 时触发：

`AttributeError: 'LazyValue' object has no attribute 'items'`

## 修复

在 EPLB 使用侧统一先解包 `LazyValue`，再执行 host mirror 初始化、异步 layer prepare、rebalance layer 列表计算和权重更新。
