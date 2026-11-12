---
id: kimi-delta-attention
leetgpuId: 120
titleEn: Kimi Delta Attention
titleZh: Kimi Delta Attention
description: 使用 CUDA 按序递推通道衰减与 Delta Rule 快速权重状态
problemUrl: https://leetgpu.com/challenges/kimi-delta-attention
updatedAt: 2026-11-12
---
## 题目定义

每个 Batch 维护 `S(d,d)`，初值为零。对每个时刻：

```text
S ← Diag(alpha[t]) S
kS = k[t]ᵀ S
S ← S + beta[t] k[t] (v[t]-kS)ᵀ
output[t] = q[t]ᵀ S
```

`alpha(batch,seq,d)` 是按 S 行通道的衰减，不能按列缩放；`beta(batch,seq)` 是标量门。Q/K/V/output 均为 `(batch,seq,d)`。K 已归一化或为零。

约束为 `batch≤8`、`seq_len≤16384`、`d≤128`，alpha 位于 `(0,1]`、beta 位于 `[0,1]`。性能用例 `4×8192×128`，容差 `1e-3`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* Q, const float* K, const float* V,
 const float* alpha, const float* beta, float* output, int batch, int seq_len, int d)
```

时间维有真实递推依赖，不能并行交换步骤。并行度来自 Batch 和状态矩阵的 d² 元素。

## 实现思路

每个 Batch 一个 Block，256 线程共同维护全局状态 S。每个时刻先并行做逐行衰减；Block 内 8 个 Warp 分摊输出列，并在 Warp 内用 Shuffle 归约 `kᵀS[:,j]`。随后 d² 状态元素并行执行 Delta 更新，最后用同样的 Warp-per-column 映射计算 `qᵀS[:,j]`。

时间循环仍按顺序，但每步最重的 `O(d²)` 工作由 256 线程并行，覆盖最大 `d=128`。列归约只需 Warp 级同步，每个时间步仅保留阶段间的 Block 屏障。状态只分配 `batch*d*d`，不随序列长度增长。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void kda_kernel(const float* Q, const float* K, const float* V,
                           const float* alpha, const float* beta, float* output,
                           float* states, int T, int d) {
    extern __shared__ float kS[];
    int b = static_cast<int>(blockIdx.x);
    int tid = static_cast<int>(threadIdx.x);
    int block_size = static_cast<int>(blockDim.x);
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps = block_size >> 5;
    float* S = states + static_cast<size_t>(b) * d * d;

    for (int t = 0; t < T; ++t) {
        size_t base = (static_cast<size_t>(b) * T + t) * d;
        for (int index = tid; index < d * d; index += block_size) {
            int row = index / d;
            S[index] *= alpha[base + row];
        }
        __syncthreads();

        for (int col = warp; col < d; col += warps) {
            float part = 0.0f;
            for (int row = lane; row < d; row += 32)
                part += K[base + row] * S[row * d + col];
            for (int off = 16; off > 0; off >>= 1)
                part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0) kS[col] = part;
        }
        __syncthreads();

        float bt = beta[static_cast<size_t>(b) * T + t];
        for (int index = tid; index < d * d; index += block_size) {
            int row = index / d, col = index % d;
            S[index] += bt * K[base + row] * (V[base + col] - kS[col]);
        }
        __syncthreads();

        for (int col = warp; col < d; col += warps) {
            float part = 0.0f;
            for (int row = lane; row < d; row += 32)
                part += Q[base + row] * S[row * d + col];
            for (int off = 16; off > 0; off >>= 1)
                part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0) output[base + col] = part;
        }
        __syncthreads();
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      const float* alpha, const float* beta, float* output,
                      int batch, int seq_len, int d) {
    float* states = nullptr;
    size_t bytes = static_cast<size_t>(batch) * d * d * sizeof(float);
    cudaMalloc(&states, bytes); cudaMemset(states, 0, bytes);
    kda_kernel<<<batch, 256, d * sizeof(float)>>>(Q, K, V, alpha, beta,
                                                          output, states, seq_len, d);
    cudaDeviceSynchronize(); cudaFree(states);
}
```
