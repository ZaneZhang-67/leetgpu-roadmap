---
id: group-normalization
leetgpuId: 105
titleEn: Group Normalization
titleZh: 组归一化
description: 使用 CUDA 按批次和连续通道组计算统计量并执行仿射归一化
problemUrl: https://leetgpu.com/challenges/group-normalization
updatedAt: 2026-11-12
---
## 题目定义

输入 `X` 和输出 `Y` 均采用 NCHW 布局，形状为 `(N,C,H,W)`。每个 batch 的 `C` 个通道分成 `G` 个连续组，每组包含 `C/G` 个通道。对每个 `(n,g)`，都在组内全部 `(C/G)*H*W` 个元素上计算总体均值和总体方差：

```text
Y[n,c,h,w] = gamma[c] * (X[n,c,h,w]-mean[n,g])
             / sqrt(var[n,g]+eps) + beta[c]
```

`G=1` 时统计范围覆盖整个样本，`G=C` 时每个通道独立统计

### 示例

```text
N=1,C=4,H=W=2,G=2,eps=1e-5
group0: channels 0/1，值为1和3，mean=2，var=1
输出对应为 -1 和 1
group1: channels 2/3，值为2和6，mean=4，var=4
输出对应为 -1 和 1
gamma=[1,1,1,1], beta=[0,0,0,0]
```

**约束与测试**

- `1≤N≤32`，`1≤C≤1024`，`1≤G≤C` 且 `C%G=0`
- `1≤H,W≤128`，`eps=1e-5`
- 输入位于 `[-100,100]`，gamma 位于 `[0.1,10]`，beta 位于 `[-10,10]`
- 测试覆盖 `G=1`、`G=C`、全零、负数、非二次幂和非方形空间维
- 性能测试为 `N=8,C=512,H=W=64,G=32`，容差 `atol=rtol=1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {}
```

官方 starter 中只有空入口。`gamma`、`beta` 是长度为 `C` 的逐通道参数，统计量按 `(batch,group)` 独立计算

## 实现思路

每个 Block 处理一个 `(n,g)`，组内元素在 NCHW 布局中连续存放。第一遍用 FP64 局部和与 Shared Memory 归约求均值；第二遍围绕已经舍入到 FP32 的均值累加 `sum((x-mean)²)`，得到稳定的总体方差。这避免 `E[x²]-E[x]²` 在“大偏置、小方差”输入上发生灾难性消减

第三遍遍历组内元素并写回结果。将局部下标除以 `H*W`，即可得到组内通道；再加上组的起始通道，就能读取对应的 gamma 和 beta。每个输出只由一个 Block 写入

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_norm_kernel(const float* X, const float* gamma,
                                  const float* beta, float* Y,
                                  int C, int H, int W, int G, float eps) {
    int ng = blockIdx.x;
    int n = ng / G, g = ng % G;
    int channels = C / G;
    size_t spatial = (size_t)H * W;
    size_t count = (size_t)channels * spatial;
    size_t base = ((size_t)n * C + (size_t)g * channels) * spatial;
    extern __shared__ double shared[];
    __shared__ float mean_value;
    __shared__ float inv_value;
    int tid = static_cast<int>(threadIdx.x);
    int block_size = static_cast<int>(blockDim.x);

    double local_sum = 0.0;
    for (size_t i = (size_t)tid; i < count; i += block_size)
        local_sum += (double)X[base + i];
    shared[tid] = local_sum;
    __syncthreads();
    for (int off = block_size / 2; off > 0; off >>= 1) {
        if (tid < off) shared[tid] += shared[tid + off];
        __syncthreads();
    }
    if (tid == 0) mean_value = (float)(shared[0] / (double)count);
    __syncthreads();

    double local_square = 0.0;
    for (size_t i = (size_t)tid; i < count; i += block_size) {
        float difference = X[base + i] - mean_value;
        local_square += (double)difference * difference;
    }
    shared[tid] = local_square;
    __syncthreads();
    for (int off = block_size / 2; off > 0; off >>= 1) {
        if (tid < off) shared[tid] += shared[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        float variance = (float)(shared[0] / (double)count);
        inv_value = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (size_t i = (size_t)tid; i < count; i += block_size) {
        int c = g * channels + (int)(i / spatial);
        Y[base + i] = (X[base + i] - mean_value) * inv_value * gamma[c] + beta[c];
    }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {
    int threads = 256;
    group_norm_kernel<<<N * G, threads, threads * sizeof(double)>>>(
        X, gamma, beta, Y, C, H, W, G, eps);
    cudaDeviceSynchronize();
}
```
