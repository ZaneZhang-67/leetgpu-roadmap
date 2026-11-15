---
id: general-matrix-multiplication-gemm
leetgpuId: 22
titleEn: General Matrix Multiplication (GEMM)
titleZh: 通用矩阵乘法
description: 使用 CUDA 计算 FP16 行主序矩阵的 C = alpha·A·B + beta·C，并用 FP32 完成累加
problemUrl: https://leetgpu.com/challenges/general-matrix-multiplication-gemm
updatedAt: 2026-11-12
---
## 题目定义

给定行主序矩阵 `A(M×K)`、`B(K×N)` 和输入输出矩阵 `C(M×N)`，计算：

```text
C = alpha × (A × B) + beta × C_initial
```

`A`、`B` 和 `C` 的元素类型均为 FP16（CUDA 中的 `half`），`alpha` 与 `beta` 是 FP32题目要求乘法累加阶段使用 FP32，最后再把结果转换为 FP16 写回 `C`

### 实现要求

- 只使用原生 CUDA 功能；题面另允许 WMMA
- `solve` 函数签名保持不变
- 乘法累加使用 FP32
- 最终结果以 `half` 写回矩阵 `C`

### 示例

```text
A（2×3） = [1 2 3]
           [4 5 6]

B（3×2） = [1 2]
           [3 4]
           [5 6]

C_initial（2×2） = [1 1]
                    [1 1]
alpha = 1.0，beta = 0.0

输出 C = [22 28]
         [49 64]
```

### 约束

- `16 ≤ M, N, K ≤ 4096`
- 性能测试使用 `M = N = K = 1024`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {}
```

平台已经准备好三个设备指针`C` 同时包含旧值并承接新结果，因此 kernel 必须先读取 `C_initial`，再覆盖同一位置函数参数顺序是 `M, N, K, alpha, beta`，参考实现保持原样

## 实现思路

### 1. 一个线程计算一个输出元素

二维线程坐标对应 `C[row, col]`：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

合法线程满足 `row < M && col < N`

### 2. 行主序索引

```text
A[row, k] → A[row * K + k]
B[k, col] → B[k * N + col]
C[row, col] → C[row * N + col]
```

矩阵 `A` 每行有 `K` 个元素，而 `B`、`C` 每行分别有 `N` 个元素，三个步长不能混用

### 3. FP32 累加与 alpha/beta

每次读取 `half` 后用 `__half2float` 转为 `float`：

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
}
```

`C_initial` 也先转为 FP32，再代入完整 GEMM 公式：

```cpp
float old_c = __half2float(C[row * N + col]);
float value = alpha * sum + beta * old_c;
C[row * N + col] = __float2half(value);
```

即使 `beta` 为零，写法仍严格保留题面语义，并覆盖任意合法的 `alpha`、`beta`

## 参考实现

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                            float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }

        int idx = row * N + col;
        float value = alpha * sum + beta * __half2float(C[idx]);
        C[idx] = __float2half(value);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);

    gemm_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
```
