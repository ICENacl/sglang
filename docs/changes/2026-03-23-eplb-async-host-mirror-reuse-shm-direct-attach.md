# eplb async host mirror reuse shm 直连优化

## 背景

`SGLANG_EPLB_ASYNC_HOST_MIRROR_REUSE_SHM=1` 的目标是复用已经存在的 `/dev/shm` expert host mirror，避免模型再次启动时重新回填整份 host mirror。

原来的实现虽然会跳过重新填充和跨节点补齐，但 `build_from_loaded_model()` 仍然会继续走一部分初始化流程，`_get_or_create_tensor()` 也会提前为所有 layer/tensor 分配 staging buffer。

这会让 `reuse_shm` 命中后仍然保留一些不必要的额外操作。

## 本次改动

- `python/sglang/srt/eplb/eplb_async_host_mirror.py`
  - 当 `reuse_shm` 命中且 valid bitmap 已完整时，`build_from_loaded_model()` 直接把 shm tensor attach 到 `layer_tensors`，不再继续走 metadata 获取、local shard 填充、跨节点补齐和完整性校验流程。
  - staging buffer 改成按需分配，只在跨节点补齐真正发生时才创建。

## 结果

- `SGLANG_EPLB_ASYNC_HOST_MIRROR_REUSE_SHM=1` 命中后，host mirror 初始化路径会直接复用 shm 中已有权重。
- 不再为这条复用路径额外做 host mirror 填充相关操作。
- 不再为不会触发远端补齐的复用路径提前分配 staging buffer。
