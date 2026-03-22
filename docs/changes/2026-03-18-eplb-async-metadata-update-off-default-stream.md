# EPLB Async 将 metadata update 从默认 stream 挪到 copy stream

## 背景

- 当前 async H2D 完成后，copy completion 线程会调用：
  - `current_metadata.update(...)`
  - `reset_async_layer_statistics(...)`
- 这两步之前没有显式指定 CUDA stream。
- 因此其中涉及 GPU tensor 的 memory op 会落到该线程的默认 stream，而不是 EPLB 的 copy stream。
- 从 nsys 上看，这些 memory op 可能排到主 stream 的 graph 之后，延长：
  - layer 的 publish 完成时间
  - `expert_location_updater.py:333` 的 host wait 时间
  - host scheduler 在 graph 之间的 bubble

## 本次修改

- 在 [expert_location_updater.py](/config/workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py) 中新增每层 `metadata_update_done_event`。
- 将以下操作显式放入 `self._copy_stream`：
  - `current_metadata.update(...)`
  - `reset_async_layer_statistics(...)`
- 在 copy stream 上 record `metadata_update_done_event`，然后由 completion 线程等待该 event 完成后，再执行：
  - `set_gpu_stage(...)`
  - 状态切换到 `PUBLISHED`

## 目的

- 避免 metadata update 占用默认 stream。
- 让 H2D、metadata publish、统计 reset 保持在同一条 EPLB stream 链上。
- 缩短主 stream graph 被这类 publish 收尾间接拖慢的概率。
