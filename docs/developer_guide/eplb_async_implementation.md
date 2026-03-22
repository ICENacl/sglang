# EPLB Async 改动记录

## 背景

- 现有 EPLB 同步模式在 rebalance 时依赖 P2P 交换 expert 权重。
- 本次改动引入 `--enable-eplb-async`，目标是在 EP 场景下通过 `/dev/shm` 中的 host mirror 和独立 CUDA stream 完成异步 H2D，并直接把权重写入 live MoE 参数。

## 主要改动

- 在 `ServerArgs` 中新增 `enable_eplb_async`，并增加约束：
  - 必须与 `--enable-eplb` 一起开启。
  - 仅支持 CUDA。
  - 与 Elastic EP、在线 `update_weights_from_disk` 互斥。
- 新增 `python/sglang/srt/eplb/eplb_async_host_mirror.py`：
  - 在模型 load 完成后，基于已经加载好的 `routed_experts_weights_of_layer` 统一构建按 layer/tensor 组织的 `/dev/shm` host mirror。
  - 每个 node 只维护一份 shared host mirror，`shm` 主体大小固定为 `num_logical_experts`，不再包含 `local_rank` 维度。
  - 每个 `local rank` 在初始化后根据自己持有的 `physical -> logical` 映射，把本地 expert 写入对应的 logical expert 槽位。
  - 对同一 node 内重复出现的 logical expert，只允许最小 `global_physical_expert_id` 的副本写入，避免 node 内重复存储。
  - 增加 layer 级 `valid bitmap`，用于标记每个 logical expert 是否已经在本 node 的 host mirror 中可用。
  - 若本 node 本地写入后仍缺失某些 logical experts，则由 `local_rank == 0` 通过跨节点传输在初始化期一次性补齐，最终让每个 node 都持有完整的 logical expert pool。
  - shm 创建逻辑改为对齐 TRT-LLM：先直接 `create=True` 创建，若遇到同名段导致 `FileExistsError`，再记录日志并执行 `close + unlink + recreate`。
  - shm 初始化时序也对齐 TRT-LLM：先由 owner 统一创建/重建所有 shm，经过一次 barrier 后，再由非 owner 进程统一 attach，避免第二次启动时 reader 抢先连接到旧 shm。
  - shm 命名前缀改为使用 `model_config.hf_config.architectures[0]` 对应的模型名，例如 `Qwen3MoeForCausalLM`，不再直接使用 `server_args.model_path`。
  - 使用 `cudaHostRegister` 让后续 H2D 可异步执行。
- 在 `FusedMoE.weight_loader` 中增加 host mirror 录制逻辑：
  - 已移除加载期即时录制逻辑，改为在模型参数加载完成后统一回填。
  - mirror 的 expert key 使用全局 logical expert id。
- 在 `ExpertLocationUpdater` 中新增 host-mirror 更新路径：
  - 保留 `same-gpu` 和 `free-rider` 优化。
  - async 路径不再分配 temp buffer，而是直接把 host mirror 权重写入 live expert 参数。
  - 每个 MoE layer 维护固定 `main_done` / `weights_ready` 事件，对齐 TRT-LLM 的主流与 balancer stream 保序协议。
  - 若 `same-gpu` 源槽位已经被本轮更早的写操作覆盖，则回退到 host mirror，避免本地交换环导致读到被覆盖的旧数据。
- 在 `FusedMoE`、`FlashInferFusedMoE`、`FlashInferFP4MoE` 的 forward 中接入层级保序：
  - 进入 layer 前等待该层最近一次异步更新完成。
  - layer 主计算和必要的 all-reduce 完成后，记录该层主流完成事件，供后续 rebalance 的 EPLB stream 等待。
- 在 `EPLBManager` 的 rebalance 日志中增加 `mode=async|sync` 标识，便于区分当前重平衡走的是异步 host-mirror 路径还是同步 P2P 路径。

## 文档与测试

- 更新 `docs/advanced_features/server_arguments.md` 和 `docs/advanced_features/expert_parallelism.md`。
- 在 `test/manual/test_expert_location_updater.py` 增加单进程 CPU 场景的 host-mirror 分支正确性测试。
- 补充本地交换环 fallback 和 free-rider 复用场景测试，覆盖直接写 live 参数后的关键边界。
- 补充 node 内唯一写者选择和多节点跨节点补齐计划的测试，覆盖 `num_logical_experts` 级 shm 布局的关键逻辑。

## 验证建议

```bash
python3 -m py_compile \
  python/sglang/srt/eplb/eplb_async_host_mirror.py \
  python/sglang/srt/eplb/expert_location_updater.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/server_args.py

python3 -m unittest test.manual.test_expert_location_updater.TestExpertLocationUpdater.test_async_host_mirror_single_rank_cpu
```
