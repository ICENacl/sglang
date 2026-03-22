# eplb async功能实现
我要实现eplb async的功能，下面关于该功能的具体描述

## eplb-sync
当前sglang已经实现eplb sync的功能，下面几点是我的简单总结，供你简单理解：
1. eplb sycn在forward结束时开始eplb任务
2. eplb sync通过p2p kernel交换权重，临时存储在temp buffer中
3. 通过d2d操作将temp buffer中交换过的权重更新到真实模型参数中

## eplb-async功能描述
我对epb-async功能有以下要求：
1. 通过server参数--enable-eplb-async控制是否启用该功能
2. eplb-async中包含3重不同层次的数据结构
   1. host-mirror。用于在host memory中存储expert权重
   2. temp buffer。用于存储eplb临时交换的权重
   3. 模型参数。模型前向推理用到的真实参数
3. 模型的moe权重在host memory中也存储一份，通过share memory存储在/dev/shm路径下，这部分可以参考trt-llm：/config/workspace/TensorRT-LLM
4. host memory中模型权重的来源请参考eplb-sync中p2p kernel使用的模型权重，存储的名称按照【模型名 + layer-id + expert-id】命令，方便取用。eplb-async只会在EP场景下使用，所以expert-id的计算需要考虑到rank信息
5. host mirror -> temp buffer：在forward结束时触发eplb-async，将host mirror中的expert权重按需copy到temp buffer中，该操作在专门的eplb stream上执行。请注意，这部分路径需要复用p2p kernel的路径，只是将p2p kernel更换成eplb steam上的h2d操作
6. temp buffer -> 模型参数。这部分路径也可以复用eplb-sync的路径，但是需要通过cuda evnet保证eplb stream上的h2d操作完成。

# eplb async功能强化
基于已实现的eplb async功能，我希望做以下功能强化：
1. 不再借助 temp buffer + D2D操作更新模型参数，直接通过H2D将host mirror中的模型权重更新到模型参数中
2. 为了保证H2D操作更新模型参数不影响主stream出现精度问题，需要做以下保序操作：
   1. 假设forward N开始EPLB，EPLB stream上的h2d操作需要wait主计算stream layer N计算完成
   2. 在forward N+1的layer N开始计算前，需要wait h2d操作计算完成