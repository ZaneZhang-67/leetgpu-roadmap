---
id: moe-top-k-gating
leetgpuId: 67
titleEn: MoE Top-K Gating
titleZh: MoE Top-K 门控
description: 使用 CUDA 为每个 token 选择降序排列的 Top-K 专家，并对选中 logits 执行数值稳定的 Softmax
problemUrl: https://leetgpu.com/challenges/moe-top-k-gating
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `[M, E]` 的 FP32 行主序矩阵 `logits`，其中 `M` 是 token 数量，`E` 是专家数量。对每一行执行：

1. 选择最大的 `k` 个 logit
2. 按 logit 降序输出对应的专家下标
3. 只对选中的 `k` 个 logit 计算 Softmax

输出形状均为 `[M, k]`：

- `topk_weights` 保存归一化权重
- `topk_indices` 保存对应的 `int32` 专家下标

每一位置的权重必须与同位置的专家下标对应

### 实现要求

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `topk_weights` 和 `topk_indices`

### 示例

```text
logits = [[1, 2, 3, 4],
          [4, 3, 2, 1]]
M = 2，E = 4，k = 2

topk_weights = [[0.7311, 0.2689],
                [0.7311, 0.2689]]
topk_indices = [[3, 2],
                [0, 1]]
```

### 约束

- `1 ≤ M ≤ 10,000`
- `1 ≤ E ≤ 256`
- `1 ≤ k ≤ E`
- logits 和权重为 FP32，下标为 `int32`
- 功能测试包含 `k = 1`、`k = E`、负 logits、`E = 64` 和 `k = 4`
- 性能测试使用 `M = 1024`、`E = 64`、`k = 2`
- 判定的绝对误差和相对误差均为 `1e-5`

官方测试使用连续随机浮点 logits，没有构造相等值。tie 情况下，`torch.topk` 的具体下标顺序不由题面单独定义。参考实现使用较小下标优先，确保结果确定；不同下标若对应值完全相等，其 Softmax 权重也相等

## 模板代码解析

CUDA starter 只提供固定入口：

```cpp
#include <cuda_runtime.h>

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(
    const float* logits,
    float* topk_weights,
    int* topk_indices,
    int M,
    int E,
    int k
) {}
```

- `logits` 是行主序的 `[M, E]` 设备矩阵
- 两个输出都是行主序的 `[M, k]` 矩阵
- `topk_indices` 的元素类型是 `int`
- starter 没有提供 kernel、临时缓冲区或启动配置

## 实现思路

一个 CUDA Block 处理一行 logits。由于 `E ≤ 256`，将该行全部加载到共享内存后，可以在 Block 内完成排序、Softmax 和写回

### 1. 加载一行

线程 `tid < E` 时读取：

```cpp
values[tid] = logits[row * E + tid];
indices[tid] = tid;
```

共享内存为固定最大容量 256，覆盖完整约束

### 2. 选出降序 Top-K

由 0 号线程对最多 256 个 `(value, index)` 对执行选择排序。每轮在未选择区间中寻找最大值，并交换到当前位置，完成前 `k` 轮即可

比较规则为：

```text
值更大者优先
值相等时下标更小者优先
```

因此前 `k` 个位置按值降序，且输出顺序确定。`k = E` 时会排列整行，`k = 1` 时只选择最大值

### 3. 数值稳定的 Softmax

先取选中值中的最大值 `values[0]`，再计算：

```text
weight[j] = exp(values[j] - values[0])
```

减去最大值可防止正 logits 导致指数溢出。最后除以选中指数项之和，使每行权重之和为 1

### 4. 写回对应关系

每个线程处理若干个 `j < k`，将 `values[j]` 对应的指数权重和 `indices[j]` 写到相同输出位置，保持位置对应关系

该实现每行选择复杂度为 `O(kE)`，在官方上限 `E = 256` 内覆盖全部约束

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void moe_topk_kernel(
    const float* logits,
    float* topk_weights,
    int* topk_indices,
    int M,
    int E,
    int k
) {
    __shared__ float values[256];
    __shared__ int indices[256];
    __shared__ float denominator;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    if (tid < E) {
        values[tid] = logits[row * E + tid];
        indices[tid] = tid;
    }
    __syncthreads();

    if (tid == 0) {
        for (int position = 0; position < k; ++position) {
            int best = position;
            for (int candidate = position + 1; candidate < E; ++candidate) {
                bool larger = values[candidate] > values[best];
                bool equalWithSmallerIndex =
                    values[candidate] == values[best]
                    && indices[candidate] < indices[best];
                if (larger || equalWithSmallerIndex) best = candidate;
            }

            float temporaryValue = values[position];
            values[position] = values[best];
            values[best] = temporaryValue;

            int temporaryIndex = indices[position];
            indices[position] = indices[best];
            indices[best] = temporaryIndex;
        }

        float maximum = values[0];
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            values[j] = expf(values[j] - maximum);
            sum += values[j];
        }
        denominator = sum;
    }
    __syncthreads();

    for (int j = tid; j < k; j += blockDim.x) {
        int outputIndex = row * k + j;
        topk_weights[outputIndex] = values[j] / denominator;
        topk_indices[outputIndex] = indices[j];
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(
    const float* logits,
    float* topk_weights,
    int* topk_indices,
    int M,
    int E,
    int k
) {
    constexpr int threadsPerBlock = 256;
    moe_topk_kernel<<<M, threadsPerBlock>>>(
        logits, topk_weights, topk_indices, M, E, k
    );
    cudaDeviceSynchronize();
}
```
