---
id: batched-matrix-multiplication
leetgpuId: 30
titleEn: Batched Matrix Multiplication
titleZh: 批量矩阵乘法
description: 使用 CUDA 对 FP32 行主序矩阵批次执行独立矩阵乘法
problemUrl: https://leetgpu.com/challenges/batched-matrix-multiplication
updatedAt: 2026-11-12
---
## 题目定义

输入 `A` 的形状为 `[BATCH,M,K]`，输入 `B` 的形状为 `[BATCH,K,N]`，计算输出 `C`，形状为 `[BATCH,M,N]`：

```text
C[b,i,j] = Σ A[b,i,k] × B[b,k,j]
```

所有矩阵均为 FP32，并按行主序连续存储。对应线性下标为：

```text
A[b,i,k] -> (b*M + i)*K + k
B[b,k,j] -> (b*K + k)*N + j
C[b,i,j] -> (b*M + i)*N + j
```

约束为 `1 ≤ BATCH ≤ 128`、`1 ≤ M,N,K ≤ 1024`。性能用例使用 `BATCH=32`、`M=N=K=256`，判定容差为 `atol = rtol = 1e-5`。

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K)
```

参数顺序是 `BATCH,M,N,K`。`challenge.py` 将 `A` 视为 `(BATCH,M,K)`，将 `B` 视为 `(BATCH,K,N)`，参考结果直接使用 `torch.bmm(A,B)` 写入 `C`，批次之间没有交叉计算。

## 实现思路

使用三维 Grid：`blockIdx.z` 选择批次，x/y 维选择输出列和行。一个线程计算一个 `C[b,row,col]`，沿 `K` 维顺序累加。

`BATCH ≤ 128`，位于 CUDA Grid 的 z 维限制内；`M,N ≤ 1024`，二维向上取整后的 Grid 也覆盖全部约束。边缘 Block 的线程先检查 `batch < BATCH && row < M && col < N`，再访问内存。

该实现不依赖方阵或示例尺寸，矩形矩阵、单批次和最大合法批次都使用同一索引公式。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C,
                                      int BATCH, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    if (batch < BATCH && row < M && col < N) {
        const float* A_batch = A + static_cast<size_t>(batch) * M * K;
        const float* B_batch = B + static_cast<size_t>(batch) * K * N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C[(static_cast<size_t>(batch) * M + row) * N + col] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                BATCH);
    batched_matmul_kernel<<<blocks, threads>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}
```
