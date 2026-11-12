---
id: softmax-attention-backward
leetgpuId: 111
titleEn: Softmax Attention Backward
titleZh: Softmax 注意力反向传播
description: 使用流式重计算与行统计计算 Softmax 注意力的 Q、K、V 梯度，避免 M×N 工作区
problemUrl: https://leetgpu.com/challenges/softmax-attention-backward
updatedAt: 2026-11-12
---
## 题目定义

`Q:[M,d]`、`K/V:[N,d]`、`dO:[M,d]`。前向没有因果 mask：

```text
P=softmax(QKᵀ/√d)
dV=PᵀdO
dP=dO Vᵀ
dS=P⊙(dP-rowsum(P⊙dP))/√d
dQ=dS K
dK=dSᵀ Q
```

### 实现要求与约束

- 保持 starter 签名，只使用 GPU 原生功能
- `1≤M,N≤100,000`，`1≤d≤128`，输入均为 FP32
- 输出分别为 `[M,d]`、`[N,d]`、`[N,d]`
- 性能测试 `M=8192,N=4096,d=128`
- 绝对误差和相对误差均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K,const float* V,const float* dO,float* dQ,float* dK,float* dV,int M,int N,int d) {}
```

`M×N` 最多达到一百亿，不能用 int 计算该乘积，也不能分配完整 score/probability 矩阵。

## 实现思路

工作区只有三个长度 M 的 double 行统计：score 最大值、指数和，以及 `Σ_j P_ij dP_ij`。每行一个 Block 并行扫描 N 得到稳定 Softmax 统计及 softmax-Jacobian 行点积。

随后每行一个 128 线程 Block 计算 dQ。每个 `(row,column)` 只由 128 个 lane 协作归约一次 score 和一次 dP；线程 `k` 随即把同一个 dS 累加到自己的 `dQ[row,k]`。每列一个 Block 以相同方式沿 M 计算 dK 与 dV。这样每个 pass 内，每个注意力元素的长度 d 点积由各 lane 协作完成，不会按输出维重复；三个 pass 的总复杂度仍为 `O(MNd)`。

工作区为 O(M) 的 double 数组，索引统一使用 `size_t`，覆盖 `M×N>INT_MAX`。三个内部 kernel 的参数都实际参与计算：row kernels 只接收 N，column kernel 只接收 M，避免未使用形参触发严格编译警告。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void row_stats(
    const float* Q, const float* K, const float* V, const float* dO,
    double* maximum, double* denominator, double* delta,
    int N, int d
) {
    __shared__ double reduction[128];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    double scale = 1.0 / sqrt(static_cast<double>(d));
    double localMaximum = -CUDART_INF;
    for (int column = 0; column < N; ++column) {
        double part = lane < d
            ? Q[size_t(row) * d + lane] * K[size_t(column) * d + lane]
            : 0.0;
        reduction[lane] = part;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        if (lane == 0) localMaximum = fmaxf(localMaximum, reduction[0] * scale);
        __syncthreads();
    }
    if (lane == 0) maximum[row] = localMaximum;
    __syncthreads();

    double localDenominator = 0.0;
    double localDeltaNumerator = 0.0;
    for (int column = 0; column < N; ++column) {
        double scorePart = lane < d
            ? Q[size_t(row) * d + lane] * K[size_t(column) * d + lane]
            : 0.0;
        reduction[lane] = scorePart;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        double score = reduction[0] * scale;
        __syncthreads();

        double dpPart = lane < d
            ? dO[size_t(row) * d + lane] * V[size_t(column) * d + lane]
            : 0.0;
        reduction[lane] = dpPart;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            double unnormalized = exp(score - localMaximum);
            localDenominator += unnormalized;
            localDeltaNumerator += unnormalized * reduction[0];
        }
        __syncthreads();
    }
    if (lane == 0) {
        denominator[row] = localDenominator;
        delta[row] = localDeltaNumerator / localDenominator;
    }
}

__global__ void grad_q(
    const float* Q, const float* K, const float* V, const float* dO,
    const double* maximum, const double* denominator, const double* delta,
    float* dQ, int N, int d
) {
    __shared__ double reduction[128];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    double scale = 1.0 / sqrt(static_cast<double>(d));
    double gradient = 0.0;
    for (int column = 0; column < N; ++column) {
        reduction[lane] = lane < d
            ? Q[size_t(row) * d + lane] * K[size_t(column) * d + lane]
            : 0.0;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        double probability = exp(reduction[0] * scale - maximum[row]) / denominator[row];
        __syncthreads();

        reduction[lane] = lane < d
            ? dO[size_t(row) * d + lane] * V[size_t(column) * d + lane]
            : 0.0;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        double dScore = probability * (reduction[0] - delta[row]) * scale;
        if (lane < d) gradient += dScore * K[size_t(column) * d + lane];
        __syncthreads();
    }
    if (lane < d) dQ[size_t(row) * d + lane] = static_cast<float>(gradient);
}

__global__ void grad_kv(
    const float* Q, const float* K, const float* V, const float* dO,
    const double* maximum, const double* denominator, const double* delta,
    float* dK, float* dV, int M, int d
) {
    __shared__ double reduction[128];
    int column = blockIdx.x;
    int lane = threadIdx.x;
    double scale = 1.0 / sqrt(static_cast<double>(d));
    double keyGradient = 0.0;
    double valueGradient = 0.0;
    for (int row = 0; row < M; ++row) {
        reduction[lane] = lane < d
            ? Q[size_t(row) * d + lane] * K[size_t(column) * d + lane]
            : 0.0;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        double probability = exp(reduction[0] * scale - maximum[row]) / denominator[row];
        __syncthreads();

        reduction[lane] = lane < d
            ? dO[size_t(row) * d + lane] * V[size_t(column) * d + lane]
            : 0.0;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        double dScore = probability * (reduction[0] - delta[row]) * scale;
        if (lane < d) {
            keyGradient += dScore * Q[size_t(row) * d + lane];
            valueGradient += probability * dO[size_t(row) * d + lane];
        }
        __syncthreads();
    }
    if (lane < d) {
        dK[size_t(column) * d + lane] = static_cast<float>(keyGradient);
        dV[size_t(column) * d + lane] = static_cast<float>(valueGradient);
    }
}

extern "C" void solve(
    const float* Q, const float* K, const float* V, const float* dO,
    float* dQ, float* dK, float* dV, int M, int N, int d
) {
    double *maximum = nullptr, *denominator = nullptr, *delta = nullptr;
    size_t bytes = static_cast<size_t>(M) * sizeof(double);
    cudaMalloc(&maximum, bytes);
    cudaMalloc(&denominator, bytes);
    cudaMalloc(&delta, bytes);
    row_stats<<<M, 128>>>(Q, K, V, dO, maximum, denominator, delta, N, d);
    grad_q<<<M, 128>>>(Q, K, V, dO, maximum, denominator, delta, dQ, N, d);
    grad_kv<<<N, 128>>>(Q, K, V, dO, maximum, denominator, delta, dK, dV, M, d);
    cudaDeviceSynchronize();
    cudaFree(delta);
    cudaFree(denominator);
    cudaFree(maximum);
}
```
