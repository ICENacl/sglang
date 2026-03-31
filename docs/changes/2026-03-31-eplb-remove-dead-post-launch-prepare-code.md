# EPLB 删除已失效的 post-launch prepare 代码

## 背景

`EPLBManager._should_use_post_launch_async_prepare()` 已经固定返回 `False`，因此 post-launch prepare 路径不会再被执行。

代码中仍残留：

- `ModelRunner.on_forward_graph_launched()` 挂点
- `EPLBManager` 里的 post-launch prepare 状态字段和辅助函数
- 只覆盖这条死路径的测试

## 修改

- 删除 `model_runner.py` 中不再使用的 `on_forward_graph_launched()` 调用和方法
- 删除 `eplb_manager.py` 中仅供 post-launch prepare 使用的状态、线程提交和快照处理逻辑
- 清理对应的单元测试，仅保留当前实际生效路径的测试

## 结果

- async EPLB 只保留当前真实生效的 prepare/apply 时序
- 代码和测试不再维护一条永远不会进入的分支
