---
id: turboquant-kv-cache-attention
leetgpuId: 95
titleEn: TurboQuant KV Cache Attention
titleZh: TurboQuant KV Cache 注意力
description: 从码本索引和 QJL 符号残差重建压缩 Key，并计算未缩放的注意力分数
problemUrl: https://leetgpu.com/challenges/turboquant-kv-cache-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q` 的形状为 `[B,D]`，压缩 Key 则由 `[S,D]` 的 `uint8 K_idx`、`int8 qjl_signs` 和 `[S]` 的 FP32 `gamma` 表示。runtime 按行向量约定执行以下计算：

```text
K_mse = codebook[K_idx] @ Pi
K_qjl = (sqrt(pi/2)/D) * gamma[:,None] * (qjl_signs @ S_mat)
K_deq = K_mse + K_qjl
scores = Q @ K_deq.T
```

输出仅包含 `[B,S]` 原始点积分数，不做 `1/sqrt(D)` 缩放或 Softmax，也不涉及 V Cache。

### 官方示例

```text
B=2,S=3,D=2,C=4, Pi=I, S_mat=I, gamma=0
Q=[[1,0],[0,1]]
K_idx=[[0,3],[1,2],[3,0]]
codebook=[-0.75,-0.25,0.25,0.75]
scores=[[-0.75,-0.25,0.75],[0.75,0.25,-0.75]]
```

### 完整约束

- `1 ≤ B ≤ 32`，`1 ≤ S ≤ 65536`，`1 ≤ D ≤ 256`，`2 ≤ C ≤ 256`
- `Pi[D,D]` 为正交矩阵；`S_mat[D,D]` 的元素来自标准正态分布
- `gamma[S]` 为 FP32；`qjl_signs[S,D]` 的 int8 值为 `-1/+1`
- `K_idx[S,D]` 为 uint8，值域 `[0,C)`；其余输入和输出为 FP32
- 性能测试为 `B=32,S=32768,D=128,C=16`
- 判定容差为 `atol=rtol=1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores are device pointers
extern "C" void solve(const float* Q, const unsigned char* K_idx,
                      const signed char* qjl_signs, const float* gamma,
                      const float* Pi, const float* S_mat,
                      const float* codebook, float* scores,
                      int B, int S, int D, int C) {}
```

`C` 只用于限定码本和索引范围，不参与矩阵尺寸；`Pi`、`S_mat` 都按 `[row,column]` 行主序存储。

## 实现思路

实现分为两个阶段：第一阶段由每个 Block 负责一个 Key 行 `s`，线程按重建后的维度 `j` 分工，分别计算：

```text
centroid_projection[j] = Σ_i codebook[K_idx[s,i]] * Pi[i,j]
sign_projection[j] = Σ_i qjl_signs[s,i] * S_mat[i,j]
```

重建结果先写入临时 `K_deq[S,D]`，这样每个 Key 只需解压一次，不必为 32 个 Batch 重复 32 次。第二阶段使用 `16×16` Shared Memory tiled GEMM 计算 `Q @ K_deqᵀ`。在性能测试规模下，复杂度从重复解压方案的 `O(B*S*D²)` 降为 `O(S*D²+B*S*D)`；QJL 系数 `sqrt(pi/2)*gamma[s]/D` 同样只需按 Key 行计算一次。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void dequantize_keys(const unsigned char* K_idx,
                                const signed char* qjl_signs,
                                const float* gamma, const float* Pi,
                                const float* S_mat, const float* codebook,
                                float* K_deq, int D) {
    int s = static_cast<int>(blockIdx.x);
    int tid = static_cast<int>(threadIdx.x);
    int block_size = static_cast<int>(blockDim.x);
    float residual_scale = sqrtf(1.5707963267948966f) * gamma[s] / D;
    size_t row = static_cast<size_t>(s) * D;
    for (int j = tid; j < D; j += block_size) {
        float mse = 0.0f;
        float residual = 0.0f;
        for (int i = 0; i < D; ++i) {
            mse += codebook[K_idx[row + i]] * Pi[i * D + j];
            residual += static_cast<float>(qjl_signs[row + i])
                        * S_mat[i * D + j];
        }
        K_deq[row + j] = mse + residual_scale * residual;
    }
}

constexpr int TILE = 16;
__global__ void score_gemm(const float* Q, const float* K_deq,
                           float* scores, int B, int S, int D) {
    __shared__ float q_tile[TILE][TILE];
    __shared__ float k_tile[TILE][TILE];
    int tx = static_cast<int>(threadIdx.x);
    int ty = static_cast<int>(threadIdx.y);
    int b = static_cast<int>(blockIdx.y) * TILE + ty;
    int s = static_cast<int>(blockIdx.x) * TILE + tx;
    float sum = 0.0f;

    for (int k0 = 0; k0 < D; k0 += TILE) {
        int k = k0 + tx;
        q_tile[ty][tx] = (b < B && k < D) ? Q[b * D + k] : 0.0f;
        int key_row = static_cast<int>(blockIdx.x) * TILE + ty;
        k_tile[ty][tx] = (key_row < S && k < D)
            ? K_deq[static_cast<size_t>(key_row) * D + k] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            sum += q_tile[ty][i] * k_tile[tx][i];
        __syncthreads();
    }
    if (b < B && s < S)
        scores[static_cast<size_t>(b) * S + s] = sum;
}

extern "C" void solve(const float* Q, const unsigned char* K_idx,
                      const signed char* qjl_signs, const float* gamma,
                      const float* Pi, const float* S_mat,
                      const float* codebook, float* scores,
                      int B, int S, int D, int C) {
    (void)C;
    float* K_deq = nullptr;
    cudaMalloc(&K_deq, static_cast<size_t>(S) * D * sizeof(float));
    dequantize_keys<<<S, 256>>>(
        K_idx, qjl_signs, gamma, Pi, S_mat, codebook, K_deq, D);
    dim3 threads(TILE, TILE);
    dim3 blocks((S + TILE - 1) / TILE, (B + TILE - 1) / TILE);
    score_gemm<<<blocks, threads>>>(Q, K_deq, scores, B, S, D);
    cudaDeviceSynchronize();
    cudaFree(K_deq);
}
```
