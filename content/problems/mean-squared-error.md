---
id: mean-squared-error
leetgpuId: 27
titleEn: Mean Squared Error
titleZh: 均方误差
description: 使用 CUDA 计算预测值与目标值之间平方误差的平均值，并处理大规模并行归约
problemUrl: https://leetgpu.com/challenges/mean-squared-error
updatedAt: 2026-11-12
---
## 题目定义

给定两个等长的 FP32 数组 `predictions` 和 `targets`，计算均方误差：

```text
MSE = (1 / N) × Σ (predictions[i] - targets[i])²
```

最终结果写入 `mse` 指向的单个 `float`

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终平均值写入 `mse`

### 示例 1

```text
predictions = [1.0, 2.0, 3.0, 4.0]
targets     = [1.5, 2.5, 3.5, 4.5]
输出：mse = 0.25
```

### 示例 2

```text
predictions = [10.0, 20.0, 30.0]
targets     = [12.0, 18.0, 33.0]
平方误差    = [4.0, 4.0, 9.0]
输出：mse = 17 / 3 ≈ 5.67
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ predictions[i], targets[i] ≤ 1000.0`
- 性能测试使用 `N = 50,000,000`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {}
```

平台已准备好三个设备指针`predictions` 和 `targets` 各有 `N` 个元素，`mse` 只有一个输出元素实现需要归约所有平方误差，并确保输出从零开始

## 实现思路

### 1. Grid-stride loop 处理大数组

`N` 最大为一亿，使用固定数量的 Block 配合 Grid-stride loop：

```cpp
for (int i = global_id; i < N; i += grid_stride) {
    float diff = predictions[i] - targets[i];
    local_sum += diff * diff;
}
```

每个线程先在寄存器中累积局部和，避免为每个元素都执行全局原子加法

### 2. Block 内归约

线程把局部和写入 Shared Memory，再用二分归约合并同一 Block 的结果每轮活跃线程数减半，并在轮次之间调用 `__syncthreads()`

Block 的线程 0 得到该 Block 的平方误差总和后，通过一次 `atomicAdd` 合并到全局输出

### 3. 平均定义

题目要求平方误差的平均值，而不是总和可以在 Block 合并时直接除以 `N`：

```cpp
atomicAdd(mse, block_sum / static_cast<float>(N));
```

`solve` 在启动 kernel 前用 `cudaMemset` 将 `mse[0]` 清零

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int N) {
    extern __shared__ float shared_sum[];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    for (int i = id; i < N; i += stride) {
        float diff = predictions[i] - targets[i];
        local_sum += diff * diff;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(mse, shared_sum[0] / static_cast<float>(N));
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    cudaMemset(mse, 0, sizeof(float));

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 4096) {
        blocks_per_grid = 4096;
    }

    mse_kernel<<<blocks_per_grid, threads_per_block,
                 threads_per_block * sizeof(float)>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
```
