---
id: multi-head-attention
leetgpuId: 12
titleEn: Multi-Head Attention
titleZh: 多头自注意力
description: 使用 CUDA 实现按特征连续分头的缩放点积自注意力，并在线完成稳定 Softmax
problemUrl: https://leetgpu.com/challenges/multi-head-attention
updatedAt: 2026-11-12
---
## 题目定义

给定三个 FP32 行主序矩阵 `Q`、`K`、`V`，形状均为 `(N, d_model)`。将最后一个维度连续切成 `h` 个头，每头宽度为：

```text
d_k = d_model / h
```

第 `head` 个头使用每行特征区间 `[head*d_k, (head+1)*d_k)`，计算：

```text
head = softmax(Q_head × K_head^T / sqrt(d_k)) × V_head
```

Softmax 沿 key 位置维度执行。各头结果写回原来的特征区间，输出形状仍为 `(N, d_model)`，没有额外输出投影。

题目约束为 `1 ≤ N ≤ 10000`、`2 ≤ d_model ≤ 1024`、`1 ≤ h ≤ d_model`、`d_model % h == 0`，输入值位于 `[-10, 10]`。性能用例为 `N = 1024`、`d_model = 1024`、`h = 16`，值采样自 `[-1, 1]`。判定容差为 `atol = rtol = 1e-5`。

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h)
```

四个数组均为设备指针。`challenge.py` 确认输入输出形状都是 `(N, d_model)`，参考语义按头切片后，对 `Q_head × K_head^T` 的每一行执行 Softmax，再乘对应的 `V_head`。

## 实现思路

一个 CUDA Block 负责一个 `(query, head)`，总计 `N*h` 个 Block。Block 内线程跨步处理该头的输出特征。为了避免申请 `N×N×h` 的分数矩阵，参考实现使用在线 Softmax：依次扫描所有 key，维护当前最大分数 `m`、归一化分母 `l` 和每个输出维度的未归一化加权和 `acc`。

当新分数 `s` 超过旧最大值时，旧累计量乘 `exp(m-s)` 后再加入新项；否则加入权重 `exp(s-m)`。扫描结束后输出 `acc/l`。这种写法与减最大值的稳定 Softmax 等价，每个 Block 只保存在线状态和一份 `d_k` 累加向量。

动态 Shared Memory 包含 256 个点积归约槽和 `d_k` 个累加值，`d_k = 1024` 时共占 5 KiB；另有 16 字节静态 Shared Memory 保存在线 Softmax 状态空间复杂度为 `O(256 + d_k)`。

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

__global__ void mha_kernel(const float* Q, const float* K, const float* V, float* output,
                           int N, int d_model, int d_k) {
    extern __shared__ float shared[];
    float* reduce = shared;
    float* acc = shared + blockDim.x;

    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int feature_base = head * d_k;

    for (int d = tid; d < d_k; d += blockDim.x) acc[d] = 0.0f;
    __shared__ float max_score;
    __shared__ float denominator;
    if (tid == 0) {
        max_score = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    float scale = rsqrtf(static_cast<float>(d_k));
    for (int key = 0; key < N; ++key) {
        float partial = 0.0f;
        for (int d = tid; d < d_k; d += blockDim.x) {
            partial += Q[query * d_model + feature_base + d] *
                       K[key * d_model + feature_base + d];
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
        for (int d = tid; d < d_k; d += blockDim.x) {
            acc[d] = acc[d] * old_scale +
                     new_weight * V[key * d_model + feature_base + d];
        }
        __syncthreads();
    }

    for (int d = tid; d < d_k; d += blockDim.x) {
        output[query * d_model + feature_base + d] = acc[d] / denominator;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
    int d_k = d_model / h;
    int threads = 256;
    dim3 grid(N, h);
    size_t shared_bytes = static_cast<size_t>(threads + d_k) * sizeof(float);
    mha_kernel<<<grid, threads, shared_bytes>>>(Q, K, V, output, N, d_model, d_k);
    cudaDeviceSynchronize();
}
```
