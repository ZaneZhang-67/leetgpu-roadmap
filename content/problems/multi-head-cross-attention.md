---
id: multi-head-cross-attention
leetgpuId: 26
titleEn: Multi-Head Cross-Attention
titleZh: 多头交叉注意力
description: 使用 CUDA 实现查询长度与键值长度不同的多头交叉注意力，并在线完成稳定 Softmax
problemUrl: https://leetgpu.com/challenges/multi-head-cross-attention
updatedAt: 2026-11-12
---
## 题目定义

查询 `Q` 的形状为 `(M, H, D)`，键 `K` 和值 `V` 的形状为 `(N, H, D)`，全部为 FP32。每个头独立计算：

```text
head_h = softmax(Q_h × K_h^T / sqrt(D)) × V_h
```

输出形状为 `(M, H, D)`。Softmax 沿 `N` 个 encoder 位置执行；没有 causal mask，每个 query 都能看到所有 key。`M` 与 `N` 可以不同。

内存按行主序连续存储，线性下标为：

```text
Q[m,h,d]      -> (m * H + h) * D + d
K[n,h,d] / V  -> (n * H + h) * D + d
output[m,h,d] -> (m * H + h) * D + d
```

约束为 `1 ≤ M,N ≤ 4096`、`1 ≤ H ≤ 64`、`1 ≤ D ≤ 256`。性能用例为 `M=1024`、`N=2048`、`H=16`、`D=128`，判定容差为 `atol = rtol = 1e-4`。

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int H, int D)
```

`challenge.py` 明确检查 `Q.shape == (M,H,D)`、`K.shape == V.shape == (N,H,D)`、`output.shape == (M,H,D)`。缩放因子是 `1/sqrt(D)`，归一化维度是最后的 key 位置维度。

## 实现思路

一个 Block 负责一个 `(query, head)`。Block 内线程协作计算当前 query 与每个 key 的长度 `D` 点积，并用在线 Softmax 同时累计 `V`。

状态包含当前最大分数 `m`、分母 `l` 和长度为 `D` 的向量 `acc`。读到新分数 `s` 时令 `m_new=max(m,s)`，再更新：

```text
l   = l * exp(m - m_new) + exp(s - m_new)
acc = acc * exp(m - m_new) + exp(s - m_new) * V[key]
```

最终输出 `acc/l`。这样无需物化大小为 `H×M×N` 的分数与注意力矩阵，覆盖性能用例时也不会产生数百 MiB 的临时数组。

`D ≤ 256`，累加向量放入动态 Shared Memory。线程数固定为 256，既能完成点积归约，也能覆盖任意合法 `D`。

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

__global__ void cross_attention_kernel(const float* Q, const float* K, const float* V,
                                       float* output, int M, int N, int H, int D) {
    extern __shared__ float shared[];
    float* reduce = shared;
    float* acc = shared + blockDim.x;

    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int q_base = (query * H + head) * D;

    for (int d = tid; d < D; d += blockDim.x) acc[d] = 0.0f;
    __shared__ float max_score;
    __shared__ float denominator;
    if (tid == 0) {
        max_score = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    float scale = rsqrtf(static_cast<float>(D));
    for (int key = 0; key < N; ++key) {
        int kv_base = (key * H + head) * D;
        float partial = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
            partial += Q[q_base + d] * K[kv_base + d];
        }
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
        for (int d = tid; d < D; d += blockDim.x) {
            acc[d] = acc[d] * old_scale + new_weight * V[kv_base + d];
        }
        __syncthreads();
    }

    for (int d = tid; d < D; d += blockDim.x) output[q_base + d] = acc[d] / denominator;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D) {
    int threads = 256;
    dim3 grid(M, H);
    size_t shared_bytes = static_cast<size_t>(threads + D) * sizeof(float);
    cross_attention_kernel<<<grid, threads, shared_bytes>>>(Q, K, V, output, M, N, H, D);
    cudaDeviceSynchronize();
}
```
