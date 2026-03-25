# EPLB async host mirror 构建时机调整

## 问题

`EPLB async` 的 host mirror 原来在 `ModelRunner.initialize()` 里同步构建。

多机下这一步可能比较慢，导致 scheduler 迟迟不能向父进程回传第一条 `ready`，从而让启动阶段的 `_launch_subprocesses` 看起来超时。

## 本次改动

- `python/sglang/srt/model_executor/model_runner.py`
  - 不再在 `initialize()` 里直接构建 host mirror
  - 增加显式方法，在模型加载完成后按需触发构建

- `python/sglang/srt/managers/scheduler.py`
  - 先向父进程发送 scheduler `ready`
  - 然后再构建 EPLB async host mirror
  - 构建完成后再进入 scheduler event loop

## 结果

- host mirror 仍然会在服务真正开始处理请求前构建完成
- 但它不再阻塞 `_launch_subprocesses` 的第一阶段 ready 握手
