# EPLB 规划前不再对 logical_count 做额外 pin_memory 拷贝

`ExpertLocationMetadata.init_by_eplb()` 之前在 CPU 算法路径里会先执行：

- `logical_count = logical_count.to("cpu")`
- `logical_count = logical_count.pin_memory()`

这里第二步会再触发一次额外 memcpy。
当上一段 D2H copy 本身已经被 CUDA stream 阻塞时，这次额外拷贝会继续把 host scheduler 卡住。

这次修改把这段逻辑收窄为：

- 只有 `async deepseek cpp` 这条真正要求 CPU tensor 的路径，才在这里提前做 `.to("cpu")`
- 其他 CPU 算法不再在这里预先搬运，沿用各自实现内部已有的 `.cpu()` 逻辑
- 去掉这里额外的 `pin_memory()`，避免再做一遍 host 侧拷贝

这样不会改变算法结果，但可以减少 host scheduler 上这段不必要的阻塞。
