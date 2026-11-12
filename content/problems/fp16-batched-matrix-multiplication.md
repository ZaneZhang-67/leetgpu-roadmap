---
id: fp16-batched-matrix-multiplication
leetgpuId: 57
titleEn: FP16 Batched Matrix Multiplication
titleZh: FP16 批量矩阵乘法
description: 对成批的 FP16 矩阵执行独立乘法，使用 FP32 累加并将最终结果转换回 half
problemUrl: https://leetgpu.com/challenges/fp16-batched-matrix-multiplication
updatedAt: 2026-11-12
---
## 题目定义

输入 `A` 的形状是 `[BATCH, M, K]`，输入 `B` 的形状是 `[BATCH, K, N]`，要求为每个批次独立计算：

```text
C[b] = A[b] × B[b]
```

输出 `C` 的形状是 `[BATCH, M, N]`所有矩阵都按行主序存储，输入和输出元素均为 FP16（CUDA 中的 `half`）乘积必须先转换成 FP32，并在 FP32 中累加，最终再转换回 FP16

### 实现要求与约束

- 不允许使用外部库
- `solve` 函数签名保持不变
- `1 ≤ BATCH ≤ 128`
- `1 ≤ M,N,K ≤ 1024`
- 性能测试使用 `BATCH=32`、`M=N=K=256`
- 官方结果先将输入转换成 FP32，完成 batched matmul 后再转成 FP16

### 示例

```text
BATCH=2, M=2, K=3, N=2
A[0] = [[1,2,3], [4,5,6]]
B[0] = [[1,2], [3,4], [5,6]]
C[0] = [[22,28], [49,64]]
```

## 模板代码解析

CUDA starter 只给出设备指针和矩阵维度：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C,
                      int BATCH, int M, int N, int K) {}
```

平台负责分配和传入 `A`、`B`、`C`，作答需要补充 GPU kernel、线程映射和启动配置`BATCH` 是批次数，参数顺序是 `BATCH, M, N, K`，不能按张量形状描述中的顺序误写

## 实现思路

用三维 Grid 表示批次、输出行和输出列：`blockIdx.z` 选择批次，二维线程坐标选择 `(row,col)`每个线程计算一个 `C[b,row,col]`

行主序下标为：

```text
A[b,row,k] = b*M*K + row*K + k
B[b,k,col] = b*K*N + k*N + col
C[b,row,col] = b*M*N + row*N + col
```

循环 `k=0..K-1`，用 `__half2float` 将两个输入转换成 `float`，乘积累加到 FP32 变量中，最后用 `__float2half` 写回这样只在最终存储时发生一次 FP16 舍入

Grid 的 `z` 维直接使用 `BATCH`，二维边界检查同时覆盖非方阵和维度不是 Block 整数倍的情况时间复杂度为 `O(BATCH·M·N·K)`，输出线程数为 `BATCH·M·N`

## 参考实现

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void batched_matmul(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (row >= M || col >= N) return;

    int aBase = batch * M * K;
    int bBase = batch * K * N;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(A[aBase + row * K + k]);
        float b = __half2float(B[bBase + k * N + col]);
        sum += a * b;
    }
    C[batch * M * N + row * N + col] = __float2half(sum);
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C,
                      int BATCH, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                BATCH);
    batched_matmul<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```
