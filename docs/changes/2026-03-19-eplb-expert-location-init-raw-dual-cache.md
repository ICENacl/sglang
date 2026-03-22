# ExpertLocationMetadata._init_raw 去除重复 D2H

本次修改优化了 `python/sglang/srt/eplb/expert_location.py` 中 `ExpertLocationMetadata._init_raw()` 的 CPU/GPU 双份 metadata 构造逻辑。

## 原问题

此前 `_init_raw()` 会直接对传入 tensor 做：

- `physical_to_logical_map.cpu()`
- `logical_to_all_physical_map_padded.cpu()`

同时 `init_by_eplb()` 还会在进入 `_init_raw()` 前先把算法输出无条件 `.to(server_args.device)`。

这会导致在 CPU 算法路径下出现以下低效往返：

1. CPU 算法先产出 CPU tensor
2. `init_by_eplb()` 先把它搬到 GPU
3. `_init_raw()` 又把它搬回 CPU，生成 `*_cpu`
4. `compute_logical_to_rank_dispatch_physical_map()` 再对 GPU map 做 host 侧访问

## 修改

- `init_by_eplb()` 不再在进入 `_init_raw()` 前无条件 `.to(server_args.device)`
- `_init_raw()` 统一负责生成：
  - pinned CPU cache
  - device cache
- 如果输入本来就是 CPU tensor：
  - 直接复用/构造 pinned CPU cache
  - device cache 从 pinned CPU cache 建立
- 如果输入本来就是 GPU tensor：
  - 只做一次 GPU -> pinned CPU 拷贝
  - 避免重复 `.cpu()`
- `compute_logical_to_rank_dispatch_physical_map()` 改为消费 CPU 侧 `logical_to_all_physical_map`
  - 最终结果再统一搬到目标 device

## 预期效果

- 去掉 `_init_raw()` 中的重复 D2H
- 去掉 CPU 算法结果“先 H2D、再 D2H”的往返
- 减少 static dispatch map 构造时对 GPU tensor 的 host 访问等待
