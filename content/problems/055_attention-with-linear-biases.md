---
id: attention-with-linear-biases
leetgpuId: 55
titleEn: Attention with Linear Biases
titleZh: 带线性偏置的注意力
description: 使用 CUDA 在线 Softmax 实现加入 alpha(i-j) 位置偏置的缩放点积注意力
problemUrl: https://leetgpu.com/challenges/attention-with-linear-biases
updatedAt: 2026-11-12
---
## 题目定义

给定 `Q(M,d)`、`K(N,d)`、`V(N,d)` 和 FP32 斜率 `alpha`，计算：

```text
scores[i,j] = dot(Q[i],K[j]) / sqrt(d) + alpha*(i-j)
output      = softmax(scores, row-wise) V
```

输出形状为 `(M,d)`。这里没有 causal mask，所有 `N` 个 key 对每个 query 都可见；`M` 与 `N` 可以不同。偏置符号必须严格使用 `i-j`，且 `alpha` 可以为负数。

正式约束为 `1≤M,N≤2048`、`1≤d≤1024`、`-1≤alpha≤1`；官方功能测试覆盖 `M≠N`、非二次幂 `d`、零矩阵和负 `alpha`；性能用例为 `M=N=2048`、`d=1024`、`alpha=0.5`，Q/K/V 位于 `[-0.1,0.1]`，判定容差为 `1e-4`

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int N, int d, float alpha)
```

四个矩阵都是 FP32 设备数组。`Q`、`K/V`、输出的行跨度均为 `d`。reference 在缩放点积之后加入 `alpha*(query-key)`，随后沿 `N` 维归一化。

## 实现思路

一个 Block 负责一个 query 行，并依次扫描所有 key。点积由 256 个线程归约，分数加入线性偏置后送入在线 Softmax；长度为 `d` 的加权值向量放在动态 Shared Memory 中。

在线更新最大值、分母和加权和，不需要物化性能用例中约 1600 万字节的分数矩阵，也避免直接指数运算溢出。`d=1024` 时动态 Shared Memory 为 `(256+1024)×4=5120` 字节。

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

__global__ void alibi_attention_kernel(const float* Q, const float* K, const float* V,
                                       float* output, int M, int N, int d, float alpha) {
    extern __shared__ float shared[];
    float* reduce = shared;
    float* acc = shared + blockDim.x;
    int query = blockIdx.x;
    int tid = threadIdx.x;
    if (query >= M) return;

    for (int feature = tid; feature < d; feature += blockDim.x) acc[feature] = 0.0f;
    __shared__ float max_score;
    __shared__ float denominator;
    if (tid == 0) { max_score = -FLT_MAX; denominator = 0.0f; }
    __syncthreads();

    float scale = rsqrtf(static_cast<float>(d));
    for (int key = 0; key < N; ++key) {
        float partial = 0.0f;
        for (int feature = tid; feature < d; feature += blockDim.x)
            partial += Q[query * d + feature] * K[key * d + feature];
        reduce[tid] = partial;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) reduce[tid] += reduce[tid + offset];
            __syncthreads();
        }

        __shared__ float old_scale;
        __shared__ float new_weight;
        if (tid == 0) {
            float score = reduce[0] * scale + alpha * static_cast<float>(query - key);
            float new_max = fmaxf(max_score, score);
            old_scale = expf(max_score - new_max);
            new_weight = expf(score - new_max);
            denominator = denominator * old_scale + new_weight;
            max_score = new_max;
        }
        __syncthreads();
        for (int feature = tid; feature < d; feature += blockDim.x)
            acc[feature] = acc[feature] * old_scale + new_weight * V[key * d + feature];
        __syncthreads();
    }

    for (int feature = tid; feature < d; feature += blockDim.x)
        output[query * d + feature] = acc[feature] / denominator;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {
    constexpr int threads = 256;
    size_t shared_bytes = static_cast<size_t>(threads + d) * sizeof(float);
    alibi_attention_kernel<<<M, threads, shared_bytes>>>(Q, K, V, output, M, N, d, alpha);
    cudaDeviceSynchronize();
}
```
