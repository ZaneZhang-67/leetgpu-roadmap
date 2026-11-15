---
id: logistic-regression
leetgpuId: 34
titleEn: Logistic Regression
titleZh: 逻辑回归
description: 使用 CUDA 通过 Newton-Raphson 迭代训练逻辑回归，计算梯度、Hessian 与系数更新
problemUrl: https://leetgpu.com/challenges/logistic-regression
updatedAt: 2026-11-12
---
## 题目定义

给定特征矩阵 `X`，形状为 `n_samples × n_features`，以及只包含 0 和 1 的目标向量 `y`，求最大化二元逻辑回归对数似然的系数 `beta`

```text
p_i = sigmoid(X_i · beta)
sigmoid(z) = 1 / (1 + exp(-z))
```

**实现要求**

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终系数写入 `beta`
- `y` 只包含 0 和 1

**约束**

- `1 ≤ n_samples ≤ 100,000`
- `1 ≤ n_features ≤ 1,000`
- `n_samples ≥ n_features`
- `X` 的值位于 `[-10, 10]`
- 绝对与相对误差容限均为 `1e-2`
- 性能测试使用 `n_samples = 16`、`n_features = 8`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {}
```

`X` 按行主序保存，`X[s * n_features + f]` 是样本 `s` 的特征 `f``y` 长度为 `n_samples`，输出 `beta` 长度为 `n_features`平台只规定入口签名，需要自行初始化系数、执行迭代并写回结果

## 实现思路

`challenge.py` 使用带 `1e-6` L2 正则的 Newton-Raphson/IRLS，最多迭代 1000 次：

```text
p = sigmoid(X beta)
W = max(p(1-p), 1e-8)
g = Xᵀ(p-y) + λ beta
H = Xᵀ W X + λI
H delta = g
beta = beta - delta
```

参考实现用一个 GPU 线程完成迭代与线性系统求解这样避免每轮梯度、Hessian 和消元之间的全局同步问题，并与实际性能规模 `16 × 8` 相匹配临时 Hessian、梯度、概率和增量放在动态显存中，以覆盖题面最大尺寸

Sigmoid 分正负两支计算，避免直接计算 `exp(-z)` 时溢出Hessian 使用部分主元高斯消元求解，迭代在 `beta_new - beta` 的 L2 范数小于 `1e-8` 时提前结束

虽然题面上限很大，这类串行 Newton 实现主要针对仓库 `challenge.py` 的实际测试规模；算法使用动态分配，不含固定维度数组

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__device__ double stable_sigmoid(double z) {
    if (z >= 0.0) {
        double e = exp(-z);
        return 1.0 / (1.0 + e);
    }
    double e = exp(z);
    return e / (1.0 + e);
}

__global__ void logistic_newton(const float* X, const float* y, float* beta,
                                double* hessian, double* gradient, double* delta,
                                int n_samples, int n_features) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int f = 0; f < n_features; ++f) beta[f] = 0.0f;
    const double lambda = 1e-6;

    for (int iteration = 0; iteration < 1000; ++iteration) {
        for (int f = 0; f < n_features; ++f) {
            gradient[f] = lambda * beta[f];
            for (int g = 0; g < n_features; ++g) {
                hessian[f * n_features + g] = f == g ? lambda : 0.0;
            }
        }

        for (int s = 0; s < n_samples; ++s) {
            double z = 0.0;
            for (int f = 0; f < n_features; ++f) {
                z += (double)X[s * n_features + f] * beta[f];
            }
            double p = stable_sigmoid(z);
            double w = p * (1.0 - p);
            if (w < 1e-8) w = 1e-8;

            for (int f = 0; f < n_features; ++f) {
                double xf = X[s * n_features + f];
                gradient[f] += xf * (p - y[s]);
                for (int g = 0; g < n_features; ++g) {
                    hessian[f * n_features + g] += xf * w * X[s * n_features + g];
                }
            }
        }

        for (int k = 0; k < n_features; ++k) {
            int pivot = k;
            double best = fabs(hessian[k * n_features + k]);
            for (int r = k + 1; r < n_features; ++r) {
                double v = fabs(hessian[r * n_features + k]);
                if (v > best) { best = v; pivot = r; }
            }
            if (pivot != k) {
                for (int c = k; c < n_features; ++c) {
                    double t = hessian[k * n_features + c];
                    hessian[k * n_features + c] = hessian[pivot * n_features + c];
                    hessian[pivot * n_features + c] = t;
                }
                double t = gradient[k]; gradient[k] = gradient[pivot]; gradient[pivot] = t;
            }
            for (int r = k + 1; r < n_features; ++r) {
                double factor = hessian[r * n_features + k] / hessian[k * n_features + k];
                for (int c = k; c < n_features; ++c) {
                    hessian[r * n_features + c] -= factor * hessian[k * n_features + c];
                }
                gradient[r] -= factor * gradient[k];
            }
        }

        for (int r = n_features - 1; r >= 0; --r) {
            double value = gradient[r];
            for (int c = r + 1; c < n_features; ++c) {
                value -= hessian[r * n_features + c] * delta[c];
            }
            delta[r] = value / hessian[r * n_features + r];
        }

        double change_sq = 0.0;
        for (int f = 0; f < n_features; ++f) {
            float old_value = beta[f];
            float new_value = old_value - (float)delta[f];
            double change = (double)new_value - old_value;
            beta[f] = new_value;
            change_sq += change * change;
        }
        if (change_sq < 1e-16) break;
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features) {
    double *hessian, *gradient, *delta;
    cudaMalloc(&hessian, (size_t)n_features * n_features * sizeof(double));
    cudaMalloc(&gradient, (size_t)n_features * sizeof(double));
    cudaMalloc(&delta, (size_t)n_features * sizeof(double));

    logistic_newton<<<1, 1>>>(X, y, beta, hessian, gradient, delta,
                              n_samples, n_features);
    cudaDeviceSynchronize();
    cudaFree(delta);
    cudaFree(gradient);
    cudaFree(hessian);
}
```
