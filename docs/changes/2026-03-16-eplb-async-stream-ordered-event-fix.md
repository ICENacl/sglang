# eplb async H2D memcpy 同步重构记录

## 改动原因

- 当前 `eplb async` 需要沿着 `docs/developer_guide/eplb_async_direct_h2d_ordering.md` 继续，切换到更接近 TRT-LLM `useGpuMemcpy` 的实现。
- 之前尝试过两条路线，但都不适合作为最终实现：
  - `weights_ready/main_done` per-layer event 路线，会在 cudagraph capture 时暴露“graph wait 图外旧 event”的问题；
  - `pending commit bank + graph 内 commit kernel` 路线复杂度过高，而且与当前 SGLang 的 direct H2D 目标不一致。
- 这次重构的目标是：
  - graph 内只保留轻量 signal；
  - graph 外由 copy stream 执行真正的 `host mirror -> live weight` H2D；
  - 同步结构对齐 TRT-LLM 的 `main -> aux -> main` 闭合链。

## 改动内容

1. `ExpertLocationUpdater`
   - 为每个 async layer 维护：
     - `signal_step_and_owner`
     - `main_to_aux_event`
     - `aux_to_main_event`
     - `copy_done_event`
   - 拆分 `aux_stream` 和 `copy_stream`：
     - `aux_stream` 只负责 graph 内 signal op；
     - `copy_stream` 只负责图外 H2D。
   - `on_forward_pass_end()` 中先让 copy stream 等待当前 forward stream，再发起 H2D，避免 live 权重与本轮尾部 compute 并发读写。
   - copy 完成后由后台线程 host-side re-arm `GPU owner`。
2. `DeepEPMoE.forward()`
   - 入口改为 `updater.wait_gpu_stage(self.layer_id)`。
   - 出口改为 `finally: updater.set_cpu_stage(self.layer_id)`，保证异常路径也会释放给 CPU stage。
3. `ModelRunner.forward()`
   - 在 `self.eplb_manager.on_forward_pass_end()` 之后，追加 `self.expert_location_updater.on_forward_pass_end()`。
4. kernel 形态
   - 删除 `sgl-kernel` 扩展中的 signal op 注册。
   - 新增 `python/sglang/jit_kernel/eplb_async_signal.py` 和 `python/sglang/jit_kernel/csrc/eplb_async_signal.cuh`。
   - `wait/set_cpu` 改为 JIT CUDA kernel，运行时通过 `load_jit(...)` 动态编译并调用。
   - `set_gpu_stage` 保持 host 侧 Python 更新，负责在 copy 完成后推进 step 并切 owner 到 GPU。
5. 文档
   - 将旧的 pending-commit 路线合并回主设计文档；
   - 主文档现在只保留当前生效的 TRT-LLM H2D memcpy 同步方案。

## 预期效果

- 避免 cudagraph capture 节点直接等待图外 stream 的旧 event，从而降低 `StreamCaptureIsolation` 风险。
- 让 replay 阶段仍然在 layer 首次读 live 权重前具备显式等待点。
- 保持 direct H2D 路线，不引入 GDRCopy 或 host-accessible live slot。
- 保持方案复杂度可控，不额外引入 TRT-LLM 的全量 worker / recapture 机制。
