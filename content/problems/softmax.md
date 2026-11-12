---
id: softmax
leetgpuId: 5
titleEn: Softmax
titleZh: Softmax 归一化
description: 使用 CUDA 计算数值稳定的 Softmax，通过最大值技巧避免指数溢出，并把输入转换为和为 1 的概率分布
problemUrl: https://leetgpu.com/challenges/softmax
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位浮点数组 `input`，在 GPU 上计算 Softmax，并把结果写入同样长度的数组 `output`

Softmax 定义为：

```text
output[i] = exp(input[i]) / Σ exp(input[j])
```

题目要求使用最大值技巧处理潜在的指数溢出先从每个输入元素中减去数组最大值，再计算指数：

```text
maxValue = max(input)
output[i] = exp(input[i] - maxValue) / Σ exp(input[j] - maxValue)
```

减去同一个常数不会改变 Softmax 的最终结果，但能保证最大的指数项为 `exp(0) = 1`

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入数组 `output`

### 示例 1

```text
输入：[1.0, 2.0, 3.0]，N = 3
输出：[0.090, 0.244, 0.665]（近似值）
```

### 示例 2

```text
输入：[-10.0, -5.0, 0.0, 5.0, 10.0]，N = 5
输出：[2.047e-09, 3.038e-07, 4.509e-05, 6.693e-03, 9.933e-01]（近似值）
```

### 约束

- `1 ≤ N ≤ 500,000`
- 性能测试使用 `N = 500,000`

## 模板代码解析

平台提供了空的 CUDA kernel 和 CPU 端启动代码：

```cpp
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

- `input` 是 GPU 上长度为 `N` 的只读数组
- `output` 是 GPU 上长度为 `N` 的输出数组
- `N` 是数组元素数量
- 模板按一个线程对应一个元素的方式计算 Grid 大小，并启动一次 `softmax_kernel`

模板只规定接口和现有启动框架，没有提供最大值、指数和或归一化逻辑这些算法步骤统一在实现思路中说明

## 实现思路

Softmax 中每个输出都依赖全局最大值和全部指数项之和，因此不能只靠彼此独立的逐元素计算完成本参考实现使用一个 Block：Block 内线程共同完成两次归约，再并行写回输出

`N` 最大为 500,000，一个 Block 的 256 个线程通过 Grid-stride 形式遍历整个数组

### 1. 归约全局最大值

每个线程先处理下标：

```text
tid, tid + 256, tid + 512, ...
```

并计算自己的局部最大值：

```cpp
float localMax = -CUDART_INF_F;
for (int i = tid; i < N; i += blockDim.x) {
    localMax = fmaxf(localMax, input[i]);
}
```

线程把局部最大值写入共享内存，再通过树形归约得到全局最大值 `shared[0]`

### 2. 使用最大值技巧计算指数

取得最大值后，每个线程再次跨步遍历输入：

```cpp
float value = expf(input[i] - maxValue);
output[i] = value;
```

这里先把未归一化的指数结果暂存到 `output`，避免后续再次计算 `expf`

例如，输入中最大值为 1000 时，直接计算 `exp(1000)` 会溢出减去最大值后，对应项变为 `exp(0)`，其余指数项不大于 1

### 3. 归约指数项之和

每个线程累加自己计算的指数值：

```cpp
float localSum = 0.0f;
for (int i = tid; i < N; i += blockDim.x) {
    localSum += output[i];
}
```

再使用共享内存树形归约，把所有局部和合并为 `sumExp`

两次归约之间都需要 `__syncthreads()`，确保共享内存完成写入后再进入下一阶段

### 4. 并行归一化

最后每个线程再次跨步处理元素：

```cpp
output[i] /= sumExp;
```

所有输出都非负，并且理论上满足：

```text
Σ output[i] = 1
```

浮点舍入可能使实际总和与 1 存在很小误差

### 5. 为什么只启动一个 Block

不同 Block 之间不能使用 `__syncthreads()` 同步若要使用多个 Block，需要拆成多个 kernel，或引入全局原子操作和额外的阶段同步

本实现使用一个 256 线程 Block，让线程通过跨步循环覆盖全部 `N` 个元素这样结构简单，并且严格保留平台要求的 `solve` 签名

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float shared[256];

    int tid = threadIdx.x;

    float localMax = -CUDART_INF_F;
    for (int i = tid; i < N; i += blockDim.x) {
        localMax = fmaxf(localMax, input[i]);
    }

    shared[tid] = localMax;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }

    float maxValue = shared[0];
    float localSum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float value = expf(input[i] - maxValue);
        output[i] = value;
        localSum += value;
    }

    shared[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    float sumExp = shared[0];
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] /= sumExp;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int threadsPerBlock = 256;
    softmax_kernel<<<1, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
