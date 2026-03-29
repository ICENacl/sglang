# EPLB Async 改为 N-1 保存输入、N launch 后继续 prepare

这次修改只调整 `enable_eplb_async` 且算法为 `deepseek/auto` 的 prepare 时序，不改 sync EPLB，也不改 async 的 layer 级 H2D / metadata publish 语义。

## 确认结果

- 仓库此前有几篇接近这个方向的设计记录。
- 但主线代码实际仍然是在 `forward N start` 的 host 主线程里直接执行 `dump_record() -> init_by_eplb()`。
- 因此“`forward N-1 end` 只保存输入，`forward N` graph launch 后再继续 prepare”的方案，在这次修改前并没有真正落地。

## 新时序

新的 async DeepSeek prepare 时序变成：

1. `forward N-1 end`
   - 只记录 rebalance boundary。
   - 冻结当时的 `physical_to_logical_map`。
   - 记录当前 stream completion event。
   - 不在这里做 `init_by_eplb()`。
2. `forward N` 的 graph / eager kernel 已经 launch 后
   - 在单独的 EPLB prepare stream 上等待上一步 event。
   - 把 buffered physical count 转成稳定的 `logical_count`。
   - 异步 D2H 到 pinned CPU。
   - 把后续 `init_by_eplb()` 提交给后台 prepare worker。
3. `forward N end`
   - 如果后台 prepare 已完成，就提交 async rebalance plan。
   - 如果还没完成，不阻塞 host，顺延到后续最早一次 `forward end` 再提交。

## 代码变化

- `python/sglang/srt/model_executor/model_runner.py`
  - 在 `_forward_raw()` 返回后新增 `on_forward_graph_launched()` 挂点。
- `python/sglang/srt/eplb/eplb_manager.py`
  - 为 async DeepSeek 路径新增 `N-1 end -> N post-launch -> N/later end apply` 三段式 prepare。
  - 新增单线程后台 prepare worker。
  - 只在结果 ready 时 apply，不再把 prepare 重算塞回 `forward start`。
- `python/sglang/srt/eplb/expert_distribution.py`
  - 新增 async rebalance snapshot。
  - 新增 detach 当前 rebalance window buffer 的能力，避免 prepare 使用旧 buffer 时和下一窗口统计互相覆盖。
- `python/sglang/jit_kernel/csrc/eplb_deepseek.cpp`
  - `rebalance_experts` 绑定释放 GIL，让后台 prepare 线程不会重新卡住 scheduler。

## 结果

- `init_by_eplb()` 的重计算不再放在 `forward start` 的 scheduler 主路径。
- 和 rebalance prepare 相关的统计快照 / D2H memory op 被挪到了 post-launch 阶段，并放到独立的 EPLB prepare stream 上。
- 如果某次 prepare 比一轮 forward 更慢，系统会延后 apply，而不是重新在 host 上同步等待。
- sync EPLB 路径不会再进入任何 async 的 `forward_pass_start/end` 或 graph-launched 处理分支。
