---
id: rms-normalization
leetgpuId: 50
titleEn: RMS Normalization
titleZh: RMS 归一化
description: 使用 CUDA 完成一维 RMSNorm 前向计算，通过平方和归约获得全局均方根，再应用标量 gamma 和 beta
problemUrl: https://leetgpu.com/challenges/rms-normalization
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的一维浮点数组 `input`，以及标量缩放参数 `gamma`、偏移参数 `beta` 和稳定项 `eps`，计算 RMS Normalization：

```text
rms = sqrt((Σ input[i]²) / N + eps)
output[i] = gamma × input[i] / rms + beta
```

RMSNorm 不减去输入均值

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
input = [1.0, 2.0, 3.0, 4.0]，N = 4
gamma = 1.0，beta = 0.0，eps = 1e-5
output ≈ [0.36514813, 0.73029625, 1.0954444, 1.4605925]
```

### 示例 2

```text
input = [1.0, 2.0, 3.0]，N = 3
gamma = 1.0，beta = 0.0，eps = 1e-5
output ≈ [0.46290955, 0.9258191, 1.3887286]
```

### 约束

- `1 ≤ N ≤ 100,000`
- `eps = 1e-5`
- `-100.0 ≤ input[i] ≤ 100.0`
- `0.1 ≤ gamma ≤ 10.0`
- `-10.0 ≤ beta ≤ 10.0`
- 功能测试包含单元素、全零、负数、非默认 `gamma` 与 `beta`，以及 `N = 2000`
- 性能测试使用 `N = 100,000`、`gamma = 1.5`、`beta = 0.0`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(
    const float* input,
    float gamma,
    float beta,
    float* output,
    int N,
    float eps
) {}
```

- `input` 和 `output` 是 GPU 指针，长度都为 `N`
- `gamma`、`beta` 和 `eps` 是按值传入的标量
- 参数顺序是 `input, gamma, beta, output, N, eps`

模板没有提供平方和缓冲区、kernel 或启动配置

## 实现思路

计算分为三个有序 GPU 阶段：先生成双精度 Block 部分和，再归约为全局平方和，最后让各线程使用同一个 RMS 写出结果

### 1. 平方和归约

每个线程先按 FP32 计算平方，再把结果提升到 FP64 做 Grid-stride 累加：

```cpp
float square = __fmul_rn(input[i], input[i]);
local += static_cast<double>(square);
```

Block 内使用双精度 Shared Memory 归约并写入部分和，第二个 kernel 再确定性归约这些部分和；非负平方项使用高精度累计，可避免多 Block FP32 原子相加在大常量输入上积累出可见误差

### 2. 计算统一的逆 RMS

输出 kernel 先把双精度总和除以 `N` 并舍入为 FP32，再按 reference 的顺序执行加 epsilon 和平方根：

```cpp
float meanSquare = float(sumSquares[0] / N);
float rms = sqrtf(meanSquare + eps);
```

`eps` 位于均值之后、平方根之内，与官方 reference 的 `sqrt(mean(input**2) + eps)` 一致

### 3. 应用缩放和偏移

每个输出元素为：

```cpp
float normalized = input[i] / rms;
output[i] = gamma * normalized + beta;
```

`gamma` 和 `beta` 是标量，不是长度为 `N` 的数组；全零输入时，`eps` 保证分母非零，结果为 `beta`

### 4. 覆盖边界

`N = 1` 时仍按同一公式计算；负输入先平方再归约，符号只保留在最终的 `input[i] / rms` 中

三个 kernel 在同一默认 stream 中依次提交，部分和、总和与输出阶段严格有序；返回前再统一同步

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void partial_squares_kernel(
    const float* input,
    double* partial,
    int N
) {
    __shared__ double shared[256];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    double local = 0.0;

    for (int i = index; i < N; i += stride) {
        float square = __fmul_rn(input[i], input[i]);
        local += static_cast<double>(square);
    }

    shared[tid] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) shared[tid] += shared[tid + step];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shared[0];
}

__global__ void finish_squares_kernel(
    const double* partial,
    double* sum_squares,
    int count
) {
    __shared__ double shared[256];
    int tid = threadIdx.x;
    double local = 0.0;
    for (int i = tid; i < count; i += blockDim.x) local += partial[i];
    shared[tid] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) shared[tid] += shared[tid + step];
        __syncthreads();
    }
    if (tid == 0) sum_squares[0] = shared[0];
}

__global__ void normalize_kernel(
    const float* input,
    const double* sum_squares,
    float gamma,
    float beta,
    float* output,
    int N,
    float eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float mean_square = __double2float_rn(sum_squares[0] / static_cast<double>(N));
    float rms = sqrtf(__fadd_rn(mean_square, eps));

    for (int i = index; i < N; i += stride) {
        float normalized = __fdiv_rn(input[i], rms);
        float scaled = __fmul_rn(gamma, normalized);
        output[i] = __fadd_rn(scaled, beta);
    }
}

// input, output are device pointers
extern "C" void solve(
    const float* input,
    float gamma,
    float beta,
    float* output,
    int N,
    float eps
) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 256) blocksPerGrid = 256;

    double* partial = nullptr;
    double* sum_squares = nullptr;
    cudaMalloc(&partial, static_cast<size_t>(blocksPerGrid) * sizeof(double));
    cudaMalloc(&sum_squares, sizeof(double));

    partial_squares_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, partial, N
    );
    finish_squares_kernel<<<1, threadsPerBlock>>>(
        partial, sum_squares, blocksPerGrid
    );
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, sum_squares, gamma, beta, output, N, eps
    );
    cudaDeviceSynchronize();

    cudaFree(sum_squares);
    cudaFree(partial);
}
```
