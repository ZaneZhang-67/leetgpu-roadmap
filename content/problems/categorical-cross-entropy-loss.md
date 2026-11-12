---
id: categorical-cross-entropy-loss
leetgpuId: 25
titleEn: Categorical Cross Entropy Loss
titleZh: 分类交叉熵损失
description: 使用 CUDA 和稳定的 log-sum-exp 公式计算批量 logits 的平均分类交叉熵
problemUrl: https://leetgpu.com/challenges/categorical-cross-entropy-loss
updatedAt: 2026-11-12
---
## 题目定义

给定行主序 logits 矩阵 `logits(N×C)` 和长度为 `N` 的真实类别数组 `true_labels`，计算批次平均分类交叉熵

单个样本 `j` 的损失为：

```text
Loss_j = log(Σ exp(logits[j,k])) - logits[j, true_labels[j]]
```

最终输出是 `N` 个样本损失的平均值：

```text
loss[0] = (Loss_0 + ... + Loss_(N-1)) / N
```

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终平均损失写入 `loss` 指向的单个 `float`

### 示例 1

```text
N = 2，C = 3
logits = [[1.0, 2.0, 0.5],
          [0.1, 3.0, 1.5]]
true_labels = [1, 1]
输出：loss = [0.3548926]
```

### 示例 2

```text
N = 3，C = 4
logits = [[-0.5, 1.5, 0.0, 1.0],
          [ 2.0,-1.0, 0.5, 0.5],
          [ 0.0, 0.0, 0.0, 0.0]]
true_labels = [3, 0, 1]
输出：loss = [0.98820376]
```

### 约束

- `1 ≤ N ≤ 10,000`
- `2 ≤ C ≤ 1,000`
- `-10.0 ≤ logits[i,j] ≤ 10.0`
- 类别下标用于访问一行中的 `C` 个元素，因此实现按 `0 ≤ true_labels[i] < C` 使用
- 性能测试使用 `N = 10,000`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {}
```

三个指针均已指向 GPU 显存`logits` 按 `N×C` 行主序存储，第 `sample` 行的起点是 `sample * C`；`loss` 只有一个元素，需要汇总所有样本后写入

## 实现思路

### 1. 使用稳定的 log-sum-exp

直接计算 `exp(logit)` 可能溢出先求一行最大值 `max_logit`，再计算：

```text
log(Σ exp(z_k)) = max_logit + log(Σ exp(z_k - max_logit))
```

单样本损失变为：

```cpp
sample_loss = max_logit + logf(exp_sum) - logits[row + label];
```

减去最大值后，指数的输入不大于零，数值更稳定

### 2. 一个线程处理一个样本

每个线程顺序扫描该样本的 `C` 个类别，先求最大值，再求指数和线程通过 Grid-stride loop 覆盖所有 `N` 个样本

不同样本的损失最终汇入同一个标量，因此用 `atomicAdd(loss, sample_loss / N)` 完成归约`solve` 必须先将 `loss[0]` 清零

### 3. 平均定义

题目要求批次平均值，而不是损失总和参考实现让每个样本在原子累加前除以 `N`：

```cpp
atomicAdd(loss, sample_loss / static_cast<float>(N));
```

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* logits, const int* true_labels, float* loss,
                                     int N, int C) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int sample = id; sample < N; sample += stride) {
        int row = sample * C;
        float max_logit = -FLT_MAX;

        for (int c = 0; c < C; ++c) {
            max_logit = fmaxf(max_logit, logits[row + c]);
        }

        float exp_sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            exp_sum += expf(logits[row + c] - max_logit);
        }

        float sample_loss = max_logit + logf(exp_sum) - logits[row + true_labels[sample]];
        atomicAdd(loss, sample_loss / static_cast<float>(N));
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 256) {
        blocks_per_grid = 256;
    }

    cross_entropy_kernel<<<blocks_per_grid, threads_per_block>>>(logits, true_labels, loss, N, C);
    cudaDeviceSynchronize();
}
```
