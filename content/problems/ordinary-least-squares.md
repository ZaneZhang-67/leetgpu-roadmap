---
id: ordinary-least-squares
leetgpuId: 33
titleEn: Ordinary Least Squares
titleZh: 普通最小二乘
description: 使用 CUDA 计算 XᵀX 与 Xᵀy，并通过带主元高斯消元求解普通最小二乘系数
problemUrl: https://leetgpu.com/challenges/ordinary-least-squares
updatedAt: 2026-11-12
---
## 题目定义

给定行主序特征矩阵 `X`，形状为 `n_samples × n_features`，以及目标向量 `y`，求使平方残差最小的系数 `beta`：

```text
min ||X beta - y||²
beta = (XᵀX)⁻¹Xᵀy
```

题目保证 `X` 满秩，因此 `XᵀX` 可逆

**实现要求**

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终系数写入 `beta`

**约束**

- `1 ≤ n_samples ≤ 100,000`
- `1 ≤ n_features ≤ 1,000`
- `n_samples ≥ n_features`
- `X` 和 `y` 的值位于 `[-1000, 1000]`
- 绝对与相对误差容限均为 `1e-2`
- 性能测试使用 `n_samples = 32`、`n_features = 32`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {}
```

`X`、`y`、`beta` 都位于 GPU 显存`X` 按样本优先的行主序布局存储，元素 `X[row, col]` 的下标为 `row * n_features + col`平台没有提供 kernel，需要自行计算正规方程并求解

## 实现思路

先构造大小为 `n_features × n_features` 的 `XᵀX` 和长度为 `n_features` 的 `Xᵀy`：

```text
XTX[r,c] = Σ X[s,r] × X[s,c]
XTy[r]   = Σ X[s,r] × y[s]
```

每个线程负责一个矩阵元素或向量元素，沿样本维度累加随后对增广矩阵 `[XTX | XTy]` 做带部分主元的高斯消元

求解阶段存在跨行依赖参考实现使用单个 GPU 线程完成消元，避免跨 Block 同步问题这覆盖最大 `n_features = 1000`，临时矩阵约含一百万个 `double`，需要动态显存而不是线程栈数组

采用 `double` 构造正规方程和消元，可以降低大量样本累加时的舍入误差最后转换为 `float` 写入 `beta`

`challenge.py` 的参考答案使用 Cholesky 分解，容差为 `1e-2`，并包含一维、正交、相关特征和最多五特征的功能测试，性能用 `32 × 32` 单位阵

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void build_normal_equations(const float* X, const float* y,
                                       double* gram, double* rhs,
                                       int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = n_features * n_features;

    if (idx < matrix_size) {
        int r = idx / n_features;
        int c = idx % n_features;
        double sum = 0.0;
        for (int s = 0; s < n_samples; ++s) {
            sum += (double)X[s * n_features + r] * X[s * n_features + c];
        }
        gram[idx] = sum;
    }

    if (idx < n_features) {
        double sum = 0.0;
        for (int s = 0; s < n_samples; ++s) {
            sum += (double)X[s * n_features + idx] * y[s];
        }
        rhs[idx] = sum;
    }
}

__global__ void solve_system(double* a, double* b, float* beta, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double best = fabs(a[k * n + k]);
        for (int r = k + 1; r < n; ++r) {
            double candidate = fabs(a[r * n + k]);
            if (candidate > best) {
                best = candidate;
                pivot = r;
            }
        }

        if (pivot != k) {
            for (int c = k; c < n; ++c) {
                double tmp = a[k * n + c];
                a[k * n + c] = a[pivot * n + c];
                a[pivot * n + c] = tmp;
            }
            double tmp = b[k];
            b[k] = b[pivot];
            b[pivot] = tmp;
        }

        double diagonal = a[k * n + k];
        for (int r = k + 1; r < n; ++r) {
            double factor = a[r * n + k] / diagonal;
            for (int c = k; c < n; ++c) {
                a[r * n + c] -= factor * a[k * n + c];
            }
            b[r] -= factor * b[k];
        }
    }

    for (int r = n - 1; r >= 0; --r) {
        double value = b[r];
        for (int c = r + 1; c < n; ++c) value -= a[r * n + c] * beta[c];
        beta[r] = (float)(value / a[r * n + r]);
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {
    double* gram;
    double* rhs;
    cudaMalloc(&gram, (size_t)n_features * n_features * sizeof(double));
    cudaMalloc(&rhs, (size_t)n_features * sizeof(double));

    int count = n_features * n_features;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    build_normal_equations<<<blocks, threads>>>(X, y, gram, rhs, n_samples, n_features);
    solve_system<<<1, 1>>>(gram, rhs, beta, n_features);
    cudaDeviceSynchronize();

    cudaFree(rhs);
    cudaFree(gram);
}
```
