---
id: causal-self-attention
leetgpuId: 53
titleEn: Causal Self-Attention
titleZh: 因果自注意力
description: 使用 CUDA 在线 Softmax 实现带上三角掩码的缩放点积自注意力
problemUrl: https://leetgpu.com/challenges/causal-self-attention
updatedAt: 2026-11-12
---
## 题目定义

给定 FP32 行主序矩阵 `Q`、`K`、`V`，形状均为 `(M,d)`，计算：

```text
output = softmax(mask(QKᵀ / sqrt(d))) V
```

对查询位置 `i`，只有 `j≤i` 的 key 可见；`j>i` 的分数等价于负无穷。Softmax 沿每行的可见 key 维执行，输出形状仍为 `(M,d)`。

正式约束为 `1≤M≤10000`、`1≤d≤128`；官方运行时性能用例为 `M=5000`、`d=128`，输入位于 `[-1,1]`，判定容差为 `atol=rtol=1e-4`；功能测试还覆盖 `d=3`、`d=5` 等非二次幂特征宽度

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d)
```

四个矩阵均按 `(M,d)` 行主序存储。reference 先计算缩放点积，屏蔽严格上三角区域，再按 `dim=1` 做 Softmax，最后右乘 `V`。

## 实现思路

一个 Block 负责一行 query。线程协作计算该 query 与 key `0..query` 的点积，未来 key 根本不读取，因此直接落实 causal mask。

为了避免分配 `M×M` 分数矩阵，使用在线稳定 Softmax。扫描新分数 `s` 时维护最大值 `m`、分母 `l` 和长度为 `d` 的加权和 `acc`：

```text
m_new = max(m,s)
l   = l*exp(m-m_new) + exp(s-m_new)
acc = acc*exp(m-m_new) + exp(s-m_new)*V[key]
```

扫描结束后输出 `acc/l`。动态 Shared Memory 保存 256 个点积归约槽和 `d` 个输出累加值。官方性能维度 `d=128`，功能用例更小；实现支持官方范围内任意正 `d`，各线程跨步处理特征。

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

__global__ void causal_attention_kernel(const float* Q, const float* K, const float* V,
                                        float* output, int M, int d) {
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
    for (int key = 0; key <= query; ++key) {
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
            float score = reduce[0] * scale;
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
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    constexpr int threads = 256;
    size_t shared_bytes = static_cast<size_t>(threads + d) * sizeof(float);
    causal_attention_kernel<<<M, threads, shared_bytes>>>(Q, K, V, output, M, d);
    cudaDeviceSynchronize();
}
```
