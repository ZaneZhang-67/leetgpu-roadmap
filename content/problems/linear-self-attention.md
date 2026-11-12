---
id: linear-self-attention
leetgpuId: 56
titleEn: Linear Self-Attention
titleZh: 线性自注意力
description: 使用 CUDA 按 ELU+1 特征映射计算线性自注意力的归一化输出
problemUrl: https://leetgpu.com/challenges/linear-self-attention
updatedAt: 2026-11-12
---
## 题目定义

给定 FP32 行主序矩阵 `Q`、`K`、`V`，形状均为 `(M,d)`。先逐元素应用特征映射：

```text
phi(x) = x+1  (x>0)
         exp(x) (x≤0)
```

再计算：

```text
S = phi(K)ᵀ V             // d×d
z = Σ_j phi(K[j])         // d
output[i] = phi(Q[i]) S / dot(phi(Q[i]),z)
```

这一定义没有 Softmax，也没有 causal mask；数学实数中的 `phi(x)` 为正；FP32 极端负值可能下溢为 0，reference 没有额外 epsilon，参考实现也不额外修正

正式约束为 `1≤M≤10000`、`1≤d≤128`；官方性能用例为 `M=10000`、`d=128`，`Q/K/V` 取值范围为 `[-100,100]`，判定容差为 `1e-4`；功能测试还覆盖零矩阵、混合正负值和非二次幂 `d`

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d)
```

四个矩阵都是 `(M,d)` 的 FP32 设备数组。reference 先构造完整 `phi_Q`、`phi_K`，再计算 `S` 和 `z`；提交代码可以省去两个中间特征矩阵，但必须保持同样的 `x>0 ? x+1 : exp(x)` 分支。

## 实现思路

先把 `K` 逐元素映射为 `phi(K)`，避免在构造统计量时反复计算指数；随后为 `S` 的每个 `(a,b)` 分配一个线程，按行顺序计算 `Σ_j phi(K[j,a])*V[j,b]`；`b=0` 的线程同时写出 `z[a]`；每个统计量只有一个写入者，不需要全局原子操作

随后由一个 Block 负责一行 query；Block 先把该行 `phi(Q)` 缓存在 Shared Memory，再按输出维度 `b` 计算 `Σ_a phi(Q[i,a])*S[a,b]`，并协作归约分母 `Σ_a phi(Q[i,a])*z[a]`

该算法不构造 `M×M` 注意力矩阵，计算量为 `O(Md²)`，工作存储为 `O(Md+d²)`；性能用例中，缓存 `phi(K)` 约占 5 MiB，`S` 只有 `128×128` 个浮点数

## 参考实现

```cpp
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float feature_map(float x) {
    return x > 0.0f ? x + 1.0f : expf(x);
}

__global__ void transform_keys(const float* K, float* phiK, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) phiK[index] = feature_map(K[index]);
}

__global__ void build_statistics(const float* phiK, const float* V,
                                 float* S, float* z, int M, int d) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * d;
    if (index >= total) return;

    int a = index / d;
    int b = index % d;
    float sum = 0.0f;
    float zSum = 0.0f;
    for (int row = 0; row < M; ++row) {
        float keyFeature = phiK[row * d + a];
        sum += keyFeature * V[row * d + b];
        if (b == 0) zSum += keyFeature;
    }
    S[index] = sum;
    if (b == 0) z[a] = zSum;
}

__global__ void produce_output(const float* Q, const float* S, const float* z,
                               float* output, int M, int d) {
    extern __shared__ float shared[];
    float* denominator_parts = shared;
    float* query_features = shared + blockDim.x;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    for (int a = tid; a < d; a += blockDim.x)
        query_features[a] = feature_map(Q[row * d + a]);
    __syncthreads();

    float local_denominator = 0.0f;
    for (int a = tid; a < d; a += blockDim.x)
        local_denominator += query_features[a] * z[a];
    denominator_parts[tid] = local_denominator;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) denominator_parts[tid] += denominator_parts[tid + offset];
        __syncthreads();
    }
    float denominator = denominator_parts[0];

    for (int b = tid; b < d; b += blockDim.x) {
        float numerator = 0.0f;
        for (int a = 0; a < d; ++a)
            numerator += query_features[a] * S[a * d + b];
        output[row * d + b] = numerator / denominator;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    float* phiK = nullptr;
    float* S = nullptr;
    float* z = nullptr;
    size_t key_bytes = static_cast<size_t>(M) * d * sizeof(float);
    size_t matrix_bytes = static_cast<size_t>(d) * d * sizeof(float);
    size_t vector_bytes = static_cast<size_t>(d) * sizeof(float);
    cudaMalloc(&phiK, key_bytes);
    cudaMalloc(&S, matrix_bytes);
    cudaMalloc(&z, vector_bytes);

    constexpr int threads = 256;
    int key_count = M * d;
    int key_blocks = (key_count + threads - 1) / threads;
    int statistics_blocks = (d * d + threads - 1) / threads;
    transform_keys<<<key_blocks, threads>>>(K, phiK, key_count);
    build_statistics<<<statistics_blocks, threads>>>(phiK, V, S, z, M, d);
    size_t shared_bytes = static_cast<size_t>(threads + d) * sizeof(float);
    produce_output<<<M, threads, shared_bytes>>>(Q, S, z, output, M, d);
    cudaDeviceSynchronize();
    cudaFree(z);
    cudaFree(S);
    cudaFree(phiK);
}
```
