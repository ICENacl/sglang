# EPLB Async Host Mirror Build 改为 tqdm 进度展示

这次改动只调整 `host mirror build` 阶段的可观测性，不改变 mirror 数据本身的构建逻辑。

## 改动内容

- `EPLBAsyncHostMirrorManager.build_from_loaded_model()` 在构建过程中增加统一的 `tqdm` 进度条
- 共享内存复用时，去掉逐个 shared memory 名称的 `info` 日志，避免 build 期间刷屏
- build 完成后，仅在 `rank 0` 打一条汇总日志

## 结果

- build 过程更适合观察整体进度，不会再被逐行日志淹没
- 完成后仍然能看到一次明确的汇总信息，包括是否复用已有数据、层数、tensor 记录数、远端补齐次数、以 GB 表示的 shared memory 总容量和耗时
