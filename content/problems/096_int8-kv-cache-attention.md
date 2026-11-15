---
id: int8-kv-cache-attention
leetgpuId: 96
titleEn: INT8 KV-Cache Attention
titleZh: INT8 KV Cache 注意力
description: 使用 CUDA 在线 Softmax 计算带逐 Token 缩放的 INT8 KV Cache 解码注意力
problemUrl: https://leetgpu.com/challenges/int8-kv-cache-attention
updatedAt: 2026-11-12
---
## 题目定义

单个新 Token 的查询 `Q(num_heads,head_dim)` 会对 INT8 Cache `K_int8`、`V_int8` 执行多头注意力。Cache 形状为 `(num_heads,seq_len,head_dim)`，每个 `(head,token)` 都有独立的 FP32 scale：

```text
K[h,s,d] = K_int8[h,s,d] × k_scale[h,s]
V[h,s,d] = V_int8[h,s,d] × v_scale[h,s]
output = softmax(QKᵀ/sqrt(head_dim))V
```

题目没有提供 causal mask 参数。查询对应当前解码 Token，需要关注给定的全部 `seq_len` 位置。

### 示例

`num_heads=1`、`seq_len=3`、`head_dim=4` 时，三个 K 向量与全 1 查询的缩放点积都是 0.5，因此 Softmax 权重均为 `1/3`，输出为 `[5,6,7,8]`。

约束为 `1≤num_heads≤64`、`1≤seq_len≤32768`、`8≤head_dim≤256` 且 head_dim 为 8 的倍数；INT8 值位于 `[-128,127]`，scale 均为正数。性能用例为 `32×8192×128`，容差为 `1e-3`。

## 模板代码解析

starter 签名为：

```text
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output,
                      int num_heads, int seq_len, int head_dim)
```

scale 按 `(num_heads,seq_len)` 布局，并非逐 head 或逐通道。K/V 的连续维度是 `head_dim`。

## 实现思路

一个 Block 负责一个 head。线程协作计算并归约每个 Token 的点积，再用数值稳定的在线 Softmax 更新分母和长度为 `head_dim` 的输出累加向量。整个过程无需分配 `num_heads×seq_len` 分数矩阵，也不会直接对大分数取指数。

扫描新分数时，旧累加乘 `exp(old_max-new_max)`，当前 V 乘 `exp(score-new_max)`。V 在使用时按当前 Token 的 `v_scale` 反量化。在最大性能用例中，32 个 Block 分别处理各个 head，每个 Block 内由 256 个线程并行处理特征维。

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kv_attention(const float* Q, const int8_t* K, const int8_t* V,
                             const float* ks, const float* vs, float* output,
                             int H, int S, int D) {
    extern __shared__ float memory[];
    float* reduce = memory;
    float* acc = memory + blockDim.x;
    int h = blockIdx.x, tid = threadIdx.x;
    if (h >= H) return;
    for (int d = tid; d < D; d += blockDim.x) acc[d] = 0.0f;
    __shared__ float maximum, denominator, old_factor, weight;
    if (tid == 0) { maximum = -FLT_MAX; denominator = 0.0f; }
    __syncthreads();
    float inv_sqrt = rsqrtf(static_cast<float>(D));

    for (int s = 0; s < S; ++s) {
        float partial = 0.0f;
        float kscale = ks[h * S + s];
        for (int d = tid; d < D; d += blockDim.x)
            partial += Q[h * D + d] * static_cast<float>(K[(h * S + s) * D + d]) * kscale;
        reduce[tid] = partial;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) reduce[tid] += reduce[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            float score = reduce[0] * inv_sqrt;
            float new_max = fmaxf(maximum, score);
            old_factor = expf(maximum - new_max);
            weight = expf(score - new_max);
            denominator = denominator * old_factor + weight;
            maximum = new_max;
        }
        __syncthreads();
        float vscale = vs[h * S + s];
        for (int d = tid; d < D; d += blockDim.x)
            acc[d] = acc[d] * old_factor
                   + weight * static_cast<float>(V[(h * S + s) * D + d]) * vscale;
        __syncthreads();
    }
    for (int d = tid; d < D; d += blockDim.x) output[h * D + d] = acc[d] / denominator;
}

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output,
                      int num_heads, int seq_len, int head_dim) {
    constexpr int threads = 256;
    kv_attention<<<num_heads, threads, (threads + head_dim) * sizeof(float)>>>(
        Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);
    cudaDeviceSynchronize();
}
```
