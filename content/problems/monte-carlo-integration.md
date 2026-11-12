---
id: monte-carlo-integration
leetgpuId: 35
titleEn: Monte Carlo Integration
titleZh: 蒙特卡洛积分
description: 使用 CUDA 归约随机采样点的函数值，并按区间宽度计算蒙特卡洛积分估计
problemUrl: https://leetgpu.com/challenges/monte-carlo-integration
updatedAt: 2026-11-12
---
## 题目定义

给定在区间 `[a, b]` 内均匀随机采样点上的函数值 `y_i = f(x_i)`，估计定积分：

```text
∫ₐᵇ f(x) dx ≈ (b - a) × (Σ y_i / n_samples)
```

随机采样点 `x_i` 已由平台处理，输入只包含函数值 `y_samples`，无需在 kernel 中生成随机数

**实现要求**

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终标量写入 `result[0]`
- 绝对与相对误差容限均为 `1e-2`

**示例**

```text
输入：a = 0，b = 2，n_samples = 8
      y_samples = [0.0625, 0.25, 0.5625, 1.0,
                   1.5625, 2.25, 3.0625, 4.0]
输出：result = 3.1875
```

**约束**

- `1 ≤ n_samples ≤ 100,000,000`
- `-1000.0 ≤ a < b ≤ 1000.0`
- 函数值范围为 `[-10000.0, 10000.0]`
- 性能测试使用 `n_samples = 10,000,000`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b,
                      int n_samples) {}
```

`y_samples` 是长度为 `n_samples` 的 GPU 浮点数组，`result` 是长度为 1 的 GPU 输出数组`a`、`b` 和 `n_samples` 按值传入平台没有提供归约 kernel 或临时空间，需要自行完成求和并写出积分估计

## 实现思路

输入最多含一亿个元素，单线程累加会成为瓶颈参考实现分两级归约：

1. 每个线程用 grid-stride 循环累加多个样本
2. 每个 Block 在共享内存中归约，写出一个部分和
3. 第二个 kernel 对部分和继续归约，直到只剩一个值
4. 最后计算 `(b-a) × sum / n_samples`

共享内存使用 `double`，减少大量正负函数值累加时的误差每轮归约启动不超过 256 个 Block，因此临时数组大小固定由首轮 Block 数决定

边界由 grid-stride 循环中的 `i < n` 保证归约 kernel 也适用于 `n_samples = 1`，最终 kernel 始终把结果写入 `result[0]`

`challenge.py` 明确检查 `y_samples.shape == (n_samples,)` 与 `result.shape == (1,)`，并包含常数函数、负函数值、窄区间和单样本测试参考公式直接使用输入均值，没有额外随机数生成步骤

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void reduce_values(const float* input, double* partial, int n) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    double sum = 0.0;

    for (; i < n; i += stride) sum += input[i];
    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shared[0];
}

__global__ void reduce_partial(const double* input, double* output, int n) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    double sum = 0.0;

    for (; i < n; i += stride) sum += input[i];
    shared[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}

__global__ void finish_integral(const double* sum, float* result,
                                float a, float b, int n_samples) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = (float)(((double)b - a) * sum[0] / n_samples);
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b,
                      int n_samples) {
    const int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;
    if (blocks > 256) blocks = 256;

    double *buffer_a, *buffer_b;
    cudaMalloc(&buffer_a, (size_t)blocks * sizeof(double));
    cudaMalloc(&buffer_b, (size_t)blocks * sizeof(double));

    reduce_values<<<blocks, threads, threads * sizeof(double)>>>(
        y_samples, buffer_a, n_samples);

    int count = blocks;
    while (count > 1) {
        int next = (count + threads - 1) / threads;
        reduce_partial<<<next, threads, threads * sizeof(double)>>>(
            buffer_a, buffer_b, count);
        double* temp = buffer_a;
        buffer_a = buffer_b;
        buffer_b = temp;
        count = next;
    }

    finish_integral<<<1, 1>>>(buffer_a, result, a, b, n_samples);
    cudaDeviceSynchronize();
    cudaFree(buffer_b);
    cudaFree(buffer_a);
}
```
