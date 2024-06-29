## ZEOR1 2 3

DeepSpeed 提供了三个级别的 ZeRO（Zero Redundancy Optimizer）优化，分别是 ZeRO-1、ZeRO-2 和 ZeRO-3。它们的主要区别如下：

1. ZeRO-1: 优化器状态分片
    - 将优化器状态（如 Adam 优化器的动量和方差）分布在多个 GPU 上。
    - 可以减少大约 4 倍的内存使用。
2. ZeRO-2: 优化器状态 + 梯度分片
    - 在 ZeRO-1 的基础上，还对梯度进行分片。
    - 可以减少大约 8 倍的内存使用。
3. ZeRO-3: 优化器状态 + 梯度 + 模型参数分片
    - 在 ZeRO-2 的基础上，还对模型参数进行分片。
    - 可以实现最大程度的内存节省，理论上可以训练无限大的模型。
    

## ZERO1

优化器状态分片是 DeepSpeed ZeRO（Zero Redundancy Optimizer）优化技术的第一个阶段（ZeRO-1）。这个概念可能看起来有点复杂，所以我们来逐步解析它：

1. 优化器状态是什么？

优化器状态指的是优化算法（如 Adam、AdamW 等）在训练过程中维护的额外信息。例如，对于 Adam 优化器，每个模型参数都对应两个状态变量：

- 一阶动量（moving average of gradients）
- 二阶动量（moving average of squared gradients）

这些状态用于调整每个参数的学习率。

1. 为什么优化器状态占用大量内存？

对于大型模型，优化器状态可能比模型参数本身占用更多的内存。例如，对于 Adam 优化器，每个模型参数都需要额外的两个浮点数来存储状态。这意味着优化器状态的内存占用是模型参数的两倍。

1. 什么是优化器状态分片？

优化器状态分片是将这些状态变量分散到多个 GPU 上的技术。而不是每个 GPU 都保存完整的优化器状态，每个 GPU 只保存一部分状态。

1. 优化器状态分片如何工作？
- 将优化器状态平均分配到所有参与训练的 GPU 上。
- 在反向传播和参数更新时，每个 GPU 只更新它负责的那部分优化器状态。
- 在需要时（如梯度更新），通过集体通信操作（如 all-gather）临时重构完整的状态。
1. 优化器状态分片的好处：
- 内存效率：每个 GPU 只需要存储一部分优化器状态，大大减少了每个 GPU 的内存需求。对于 N 个 GPU，理论上可以减少 N 倍的优化器状态内存使用。
- 可扩展性：允许训练更大的模型或使用更大的批次大小。
- 计算效率：虽然引入了一些通信开销，但通常可以通过更大的批次大小或模型规模来抵消。
1. 优化器状态分片的权衡：
- 增加了一些通信开销，因为 GPU 之间需要交换信息来重构完整的优化器状态。
- 实现复杂性增加，需要特殊的训练框架支持（如 DeepSpeed）。

## zero2

ZeRO-2（Zero Redundancy Optimizer 第二阶段）在 ZeRO-1 的基础上增加了梯度分片。让我们深入解析这个概念：

1. 回顾 ZeRO-1：
ZeRO-1 实现了优化器状态的分片，减少了每个 GPU 上优化器状态的内存占用。
2. ZeRO-2 的额外优化 - 梯度分片：
梯度分片指的是将模型参数的梯度分散到多个 GPU 上，而不是每个 GPU 都保存完整的梯度副本。
3. 梯度在训练中的角色：
    - 在反向传播过程中，计算每个参数相对于损失函数的梯度。
    - 这些梯度用于更新模型参数。
    - 在分布式训练中，通常需要在所有 GPU 上对梯度进行平均。
4. ZeRO-2 如何实现梯度分片：
    - 每个 GPU 只负责计算和存储整个模型梯度的一部分。
    - 在需要完整梯度时（如参数更新前），通过集体通信操作（如 all-gather）临时重构完整的梯度。
5. ZeRO-2 的工作流程：
a. 前向传播：正常进行。
b. 反向传播：每个 GPU 只计算分配给它的那部分参数的梯度。
c. 梯度聚合：使用 reduce-scatter 操作，每个 GPU 获得它负责的那部分参数的完整梯度。
d. 参数更新：每个 GPU 只更新它负责的那部分参数。
6. ZeRO-2 的优势：
    - 进一步减少内存使用：相比 ZeRO-1，ZeRO-2 可以额外减少约 2 倍的内存使用。
    - 允许训练更大的模型：通过减少每个 GPU 上的内存占用，可以处理更大的模型或使用更大的批次大小。
    - 保持计算效率：尽管增加了一些通信开销，但通常可以通过更大的模型或批次大小来抵消。
7. ZeRO-2 的权衡：
    - 增加了通信开销：需要更多的集体通信操作来重构完整的梯度。
    - 实现复杂性更高：需要更复杂的训练框架支持。
8. ZeRO-2 vs ZeRO-1：
    - ZeRO-2 在内存效率上优于 ZeRO-1，因为它同时分片了优化器状态和梯度。
    - ZeRO-2 允许训练更大的模型，但可能会有稍高的通信开销。

总的来说，ZeRO-2 通过同时分片优化器状态和梯度，提供了比 ZeRO-1 更激进的内存优化。这使得在有限的 GPU 内存下可以训练更大的模型或使用更大的批次大小，是一种在大规模分布式训练中非常有效的内存优化技术。

让我详细解释一下 ZeRO-2 中参数更新和整合的过程：

1. 参数更新过程：
a. 每个 GPU 只更新它负责的那部分参数。
b. 更新完成后，每个 GPU 上都只有部分更新后的模型参数。
2. 参数整合：
实际上，ZeRO-2 并不会在每次更新后立即整合所有参数到每个 GPU 上。相反，它采用了一种"按需"的方式来整合参数。
3. 按需整合的工作原理：
a. 前向传播时：

b. 反向传播时：

c. 优化器步骤：
    - 当某个 GPU 需要使用不在自己负责范围内的参数时，它会向拥有该参数的 GPU 请求。
    - 使用集体通信操作（如 all-gather）来获取所需的参数。
    - 类似前向传播，需要时才获取其他 GPU 上的参数。
    - 每个 GPU 只更新它负责的那部分参数，不需要整合。
4. 通信优化：
    - 为了减少通信开销，ZeRO-2 实现了一些优化技术，如通信和计算重叠、bucket 通信等。
5. 模型保存和加载：
    - 保存模型时，需要收集所有 GPU 上的参数。
    - 加载模型时，会将参数再次分配到各个 GPU 上。
6. 优点：
    - 减少了不必要的通信：只在实际需要时才传输参数。
    - 保持了内存效率：每个 GPU 大部分时间只保存部分参数。
7. 权衡：
    - 增加了一些即时通信开销：在需要使用其他 GPU 上的参数时，需要额外的通信。
    - 实现复杂性增加：需要仔细管理参数的位置和通信。
8. 与 ZeRO-1 的区别：
    - ZeRO-1 只分片优化器状态，模型参数在每个 GPU 上都有完整副本。
    - ZeRO-2 分片了优化器状态和梯度，同时也分片了模型参数，但通过按需通信来保证计算正确性。

总的来说，ZeRO-2 通过这种"分而治之"的方法，实现了更高的内存效率，同时通过智能的通信策略保持了计算效率。这种方法不需要在每次参数更新后立即整合所有参数，而是在需要时动态整合，从而在内存使用和计算效率之间取得了很好的平衡。

### zero2 和 zero3 模型分片的区别

虽然 ZeRO-2 和 ZeRO-3 看起来都涉及参数分片，但它们在处理参数的方式上有重要的区别。让我们来详细比较一下 ZeRO-2 和 ZeRO-3：

ZeRO-2：

1. 分片内容：
    - 优化器状态
    - 梯度
    - 参数（部分分片）
2. 参数处理：
    - 在大多数时间保持完整的参数副本
    - 仅在内存压力大时才进行参数分片
    - 需要时快速重构完整参数
3. 内存效率：
    - 比 ZeRO-1 更节省内存，但不如 ZeRO-3
4. 通信开销：
    - 相对较低，主要在梯度聚合和偶尔的参数重构
5. 实现复杂度：
    - 中等

ZeRO-3：

1. 分片内容：
    - 优化器状态
    - 梯度
    - 参数（完全分片）
2. 参数处理：
    - 始终保持参数完全分片
    - 在计算过程中动态重构需要的参数
    - 更频繁的参数通信
3. 内存效率：
    - 最高的内存效率
    - 理论上可以训练无限大的模型（受总 GPU 内存限制）
4. 通信开销：
    - 较高，需要频繁的参数通信
    - 实现了更复杂的通信优化策略
5. 实现复杂度：
    - 最高

主要区别：

1. 参数分片程度：
    - ZeRO-2：参数分片是可选的，主要用于处理内存压力
    - ZeRO-3：参数始终完全分片across所有 GPU
2. 内存效率：
    - ZeRO-3 提供了最极致的内存优化，适合超大模型
    - ZeRO-2 在内存效率和通信开销之间取得了平衡
3. 通信模式：
    - ZeRO-2：主要通信发生在梯度聚合阶段
    - ZeRO-3：在前向和反向传播过程中都需要频繁通信参数
4. 适用场景：
    - ZeRO-2：适合大多数大规模分布式训练场景
    - ZeRO-3：特别适合超大模型，或者在非常有限的 GPU 内存下训练大模型
5. 实现复杂度：
    - ZeRO-3 需要更复杂的实现，对训练框架的要求更高
    

## zero1 示例

{
"zero_optimization": {
"stage": 1,
"reduce_bucket_size": 5e8,
"allgather_bucket_size": 5e8
},
"optimizer": {
"type": "Adam",
"params": {
"lr": 1e-5,
"betas": [0.9, 0.999],
"eps": 1e-8
}
},
"scheduler": {
"type": "WarmupLR",
"params": {
"warmup_min_lr": 0,
"warmup_max_lr": 1e-5,
"warmup_num_steps": 100
}
}
}

在 DeepSpeed ZeRO-1 中,`reduce_bucket_size` 和 `allgather_bucket_size` 是两个重要的参数,它们与梯度通信和内存优化有关。

1. `reduce_bucket_size`:
    - 在分布式训练中,每个进程（或GPU）计算出梯度后,需要进行梯度归约(Gradient Reduction),即将所有进程的梯度相加,得到最终的梯度更新。
    - `reduce_bucket_size` 参数定义了在进行梯度归约时,每个桶(bucket)的大小。桶是一种用于批量化梯度通信的机制,可以减少通信次数和延迟。
    - 设置较大的 `reduce_bucket_size` 可以减少梯度通信的次数,提高通信效率。但是,过大的桶大小可能会增加内存占用。
2. `allgather_bucket_size`:
    - 在 ZeRO-1 中,优化器状态（如动量和梯度平方）是分片存储的,每个进程只保存一部分。
    - 在进行优化器步骤时,需要将分片的优化器状态聚合(AllGather)起来,以便每个进程都有完整的优化器状态信息。
    - `allgather_bucket_size` 参数定义了在进行优化器状态聚合时,每个桶的大小。与梯度归约类似,使用桶可以批量化优化器状态的通信。
    - 设置较大的 `allgather_bucket_size` 可以减少优化器状态通信的次数,提高通信效率。但是,过大的桶大小可能会增加内存占用。

这两个参数的设置需要权衡通信效率和内存占用。较大的桶大小可以减少通信次数,提高效率,但会增加内存占用。较小的桶大小可以减少内存占用,但会增加通信次数,降低效率。

一般来说,可以根据模型的大小、可用内存以及网络带宽等因素来调整这两个参数。DeepSpeed 提供了一些默认值,如 `reduce_bucket_size` 和 `allgather_bucket_size` 都设置为 5e8（500MB）,这在大多数情况下都能提供较好的性能。但对于特别大或特别小的模型,你可能需要根据实际情况进行调整。

总之,`reduce_bucket_size` 和 `allgather_bucket_size` 参数通过批量化梯度和优化器状态的通信,可以提高 ZeRO-1 的通信效率,同时需要权衡内存占用。合适的参数设置可以帮助你在分布式训练中获得更好的性能。

1. `"lr"` (Learning Rate):
    - 学习率,控制每次参数更新的步长。
    - 在这个例子中,学习率被设置为 1e-5,即 0.00001。
    - 学习率是一个重要的超参数,它决定了模型参数更新的速度。较大的学习率可以加快收敛,但可能导致不稳定;较小的学习率可以稳定训练,但收敛速度较慢。
2. `"betas"` (Beta coefficients):
    - Adam 优化器使用两个动量项来自适应地调整每个参数的学习率。`"betas"` 参数是一个包含两个浮点数的列表,表示这两个动量项的衰减率。
    - 第一个 beta 值（通常称为 beta1）控制梯度的一阶矩（即梯度的移动平均）的衰减率。通常设置为接近 1 的值,如 0.9。
    - 第二个 beta 值（通常称为 beta2）控制梯度的二阶矩（即梯度平方的移动平均）的衰减率。通常设置为接近 1 的值,如 0.999。
    - 这些 beta 值影响了优化器对梯度历史信息的考虑程度。较大的 beta 值会给予过去的梯度更多的权重,而较小的 beta 值则更关注当前的梯度。
3. `"eps"` (Epsilon):
    - Adam 优化器中用于提高数值稳定性的小常数。
    - 在计算梯度的自适应学习率时,添加一个小的 epsilon 值可以防止除以零的情况。
    - 通常设置为一个很小的正数,如 1e-8。
    

## zero2 示例

{
"fp16": {
"enabled": "auto",
"loss_scale": 0,
"loss_scale_window": 1000,
"initial_scale_power": 16,
"hysteresis": 2,
"min_loss_scale": 1
},
"optimizer": {
"type": "AdamW",
"params": {
"lr": "auto",
"betas": "auto",
"eps": "auto",
"weight_decay": "auto"
}
},

```
"scheduler": {
    "type": "WarmupDecayLR",
    "params": {
        "last_batch_iteration": -1,
        "total_num_steps": "auto",
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto"
    }
},

"zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
    },
    "offload_param": {
        "device": "cpu",
        "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
},
"activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
},
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"steps_per_print": 2000,
"train_batch_size": "auto",
"min_lr": 5e-7,
"train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}
```

1. `"fp16"`: 控制半精度(FP16)训练的设置。
    - `"enabled"`: 设置为 "auto",表示自动启用 FP16 训练。
    - `"loss_scale"`: 损失缩放因子,用于防止 FP16 训练时的梯度下溢。设置为 0 表示自动选择缩放因子。
    - `"loss_scale_window"`: 自动损失缩放的窗口大小。
    - `"initial_scale_power"`: 初始损失缩放因子的指数。
    - `"hysteresis"`: 自动损失缩放的滞后因子。
    - `"min_loss_scale"`: 最小损失缩放因子。
2. `"optimizer"`: 优化器的设置。
    - `"type"`: 优化器类型,这里使用 AdamW 优化器。
    - `"params"`: 优化器的超参数。
        - `"lr"`: 学习率,设置为 "auto" 表示自动选择学习率。
        - `"betas"`: Adam 优化器的两个动量衰减率,设置为 "auto" 表示自动选择。
        - `"eps"`: 优化器的 epsilon 值,设置为 "auto" 表示自动选择。
        - `"weight_decay"`: L2 正则化的权重衰减系数,设置为 "auto" 表示自动选择。
3. `"scheduler"`: 学习率调度器的设置。
    - `"type"`: 学习率调度器类型,这里使用 WarmupDecayLR 调度器。
    - `"params"`: 学习率调度器的参数。
        - `"last_batch_iteration"`: 上一次迭代的批次数,设置为 -1 表示从头开始训练。
        - `"total_num_steps"`: 总的训练步数,设置为 "auto" 表示自动计算。
        - `"warmup_min_lr"`: 预热期间的最小学习率,设置为 "auto" 表示自动选择。
        - `"warmup_max_lr"`: 预热期间的最大学习率,设置为 "auto" 表示自动选择。
        - `"warmup_num_steps"`: 预热的步数,设置为 "auto" 表示自动计算。
4. `"zero_optimization"`: ZeRO 优化的设置。
    - `"stage"`: ZeRO 优化的阶段,这里设置为 2,表示使用 ZeRO-2 优化。
    - `"offload_optimizer"`: 优化器状态的卸载设置。
        - `"device"`: 卸载的设备,这里设置为 "cpu"。
        - `"pin_memory"`: 是否将卸载的优化器状态固定在内存中,设置为 true。
    - `"offload_param"`: 模型参数的卸载设置,类似于优化器状态的卸载设置。
    - `"allgather_partitions"`: 是否在 ZeRO-2 中使用 allgather 进行梯度聚合,设置为 true。
    - `"allgather_bucket_size"`: allgather 操作的桶大小,设置为 5e8 字节。
    - `"overlap_comm"`: 是否重叠通信和计算,设置为 true。
    - `"reduce_scatter"`: 是否使用 reduce-scatter 进行梯度聚合,设置为 true。
    - `"reduce_bucket_size"`: reduce-scatter 操作的桶大小,设置为 5e8 字节。
    - `"contiguous_gradients"`: 是否使用连续的梯度缓冲区,设置为 true。
5. `"activation_checkpointing"`: 激活检查点的设置,用于减少内存占用。
    - `"partition_activations"`: 是否对激活进行分区,设置为 false。
    - `"cpu_checkpointing"`: 是否在 CPU 上存储检查点,设置为 false。
    - `"contiguous_memory_optimization"`: 是否使用连续内存优化,设置为 false。
    - `"number_checkpoints"`: 检查点的数量,设置为 null 表示自动选择。
    - `"synchronize_checkpoint_boundary"`: 是否在检查点边界处同步,设置为 false。
    - `"profile"`: 是否对检查点进行性能分析,设置为 false。
6. 其他设置:
    - `"gradient_accumulation_steps"`: 梯度累积的步数,设置为 "auto" 表示自动选择。
    - `"gradient_clipping"`: 梯度裁剪的阈值,设置为 "auto" 表示自动选择。
    - `"steps_per_print"`: 每隔多少步打印一次训练信息,这里设置为 2000。
    - `"train_batch_size"`: 训练的批次大小,设置为 "auto" 表示自动选择。
    - `"min_lr"`: 最小学习率,这里设置为 5e-7。
    - `"train_micro_batch_size_per_gpu"`: 每个 GPU 上的微批次大小,设置为 "auto" 表示自动选择。
    - `"wall_clock_breakdown"`: 是否打印每个训练步骤的时间细分,设置为 false。
    1. "offload_optimizer" 和 "offload_param": 这两个参数与 ZeRO-Offload 有关,它是 ZeRO-2 的一个扩展,允许将优化器状态和模型参数从 GPU 内存卸载到 CPU 内存或 NVMe 设备。这有助于进一步减少 GPU 内存的使用量。
        - "device": 指定卸载的目标设备,可以是 "cpu" 或 "nvme"。
        - "pin_memory": 如果设置为 true,将卸载到 CPU 的优化器状态固定在内存中,提高数据传输效率。
    2. "allgather_partitions" 和 "allgather_bucket_size": 这两个参数与 ZeRO-2 中的梯度聚合有关。在 ZeRO-2 中,梯度被分割成多个分区,每个分区在不同的 GPU 上计算。allgather 操作用于将这些分区聚合起来形成完整的梯度。
        - "allgather_partitions": 如果设置为 true,启用 allgather 进行梯度聚合。
        - "allgather_bucket_size": 指定 allgather 操作的桶大小（以字节为单位）。较大的桶大小可以提高通信效率,但会消耗更多内存。
    3. "overlap_comm": 如果设置为 true,DeepSpeed 将尝试重叠通信和计算,以提高训练效率。这意味着在等待梯度聚合完成的同时,模型可以继续进行前向和反向传播计算。
    4. "reduce_scatter" 和 "reduce_bucket_size": 这两个参数与另一种梯度聚合方式 reduce-scatter 有关。与 allgather 不同,reduce-scatter 在聚合梯度的同时也进行了梯度的归约。
        - "reduce_scatter": 如果设置为 true,启用 reduce-scatter 进行梯度聚合和归约。
        - "reduce_bucket_size": 指定 reduce-scatter 操作的桶大小（以字节为单位）。
    5. "contiguous_gradients": 如果设置为 true,DeepSpeed 将使用连续的内存缓冲区来存储梯度,这可以提高内存访问效率和通信性能。

### 梯度下溢的原因

在 FP16（半精度）训练中,梯度下溢是一个常见的问题。这是由于 FP16 的有限表示范围和精度导致的。

FP16 的表示范围大约是 6x10^-5 到 6x10^4,而梯度值通常很小,特别是在深度神经网络的后层。当这些小梯度与损失函数的值相乘时,结果可能会低于 FP16 的最小表示范围,从而导致梯度下溢。

梯度下溢会导致以下问题:

1. 梯度变为零: 当梯度值太小时,它们可能会被四舍五入为零。这意味着该参数将不会得到更新,从而停止学习。
2. 精度损失: 即使梯度没有完全下溢到零,由于 FP16 的精度限制,梯度的精度也可能大大降低。这可能导致训练不稳定或收敛速度变慢。

为了解决梯度下溢的问题,DeepSpeed 和其他 FP16 训练框架使用了一种称为"损失缩放"（Loss Scaling）的技术。损失缩放的基本思想是在计算损失和梯度之前,将损失值乘以一个比例因子（通常是 2 的幂）。这样可以将损失和梯度值的数量级提高到 FP16 的表示范围内,从而防止梯度下溢。

例如,如果原始损失为 0.001,梯度为 1e-5,乘以一个 1024 的缩放因子后,损失变为 1.024,梯度变为 0.0102,这样就可以在 FP16 中正确表示。

在反向传播后,DeepSpeed 会将梯度除以相同的缩放因子,以恢复原始的梯度值。这个过程称为"损失缩放反向"（Loss Unscaling）。

DeepSpeed 支持自动损失缩放,即根据训练过程中的梯度情况自动调整缩放因子。当设置 `"loss_scale": 0` 时,DeepSpeed 会自动选择最优的缩放因子,以最大限度地防止梯度下溢,同时保持训练的稳定性。

总之,FP16 训练中的梯度下溢问题是由 FP16 的有限表示范围和精度引起的。DeepSpeed 通过损失缩放技术有效地解决了这个问题,使得 FP16 训练能够稳定、高效地进行。