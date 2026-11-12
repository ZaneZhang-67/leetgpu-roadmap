---
id: int8-quantized-matmul
leetgpuId: 32
titleEn: INT8 Quantized MatMul
titleZh: INT8 量化矩阵乘法
description: 使用 CUDA 完成带 scale 与 zero-point 的 INT8 行主序矩阵乘法、偶数舍入和饱和截断
problemUrl: https://leetgpu.com/challenges/int8-quantized-matmul
updatedAt: 2026-11-12
---
## 题目定义

给定 INT8 行主序矩阵 `A(M×K)`、`B(K×N)`，三个量化 scale 和三个 zero-point，计算 INT8 输出 `C(M×N)`：

```text
acc = Σ (A[i,k] - zero_point_A) × (B[k,j] - zero_point_B)
q   = round_to_even(acc × scale_A × scale_B / scale_C) + zero_point_C
C   = clamp(q, -128, 127)
```

累加使用 INT32，缩放使用 FP32。舍入必须是 round-half-to-even：`2.5 -> 2`，`-3.5 -> -4`。CUDA 中应使用 `rintf`，不能换成 ties-away-from-zero 的 `roundf`。

约束为 `1 ≤ M,N,K ≤ 4096`；三个 scale 均为正浮点数；三个 zero-point 位于 `[-128,127]`。`challenge.py` 的性能用例实际使用 `M=8192`、`N=4096`、`K=2048`，因此参考实现按更大 `M` 同样有效。输出要求逐元素完全一致，`atol = rtol = 0`。

## 模板代码解析

CUDA starter 的真实签名为：

```text
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C,
                      int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C)
```

`A`、`B`、`C` 是设备指针。矩阵按行主序解释：`A[i,k]` 下标为 `i*K+k`，`B[k,j]` 下标为 `k*N+j`。题面要求 INT32 点积；当前 `challenge.py` 将修正后的整数转成 FP32 做矩阵乘法，再先舍入为整数，用它近似该累加结果随后按顺序应用三个 scale、执行偶数舍入、加输出 zero-point，最后截断到 INT8 范围。

## 实现思路

一个线程计算一个输出元素。输入值先提升为 `int`，再减 zero-point 并相乘；在最大 `K=4096` 时，单项绝对值不超过 `255²`，总和不超过约 2.66 亿，处于 INT32 范围内。

缩放必须保留 reference 的 FP32 运算顺序：

```text
scaled = float(acc)
scaled = scaled * scale_A
scaled = scaled * scale_B
scaled = scaled / scale_C
```

预先合并 multiplier 会改变中间舍入，在半整数边界上可能让最终量化值相差 1参考实现使用 `__fmul_rn` 与 `__fdiv_rn` 固定每一步的单精度舍入，再用 `rintf(scaled)` 执行 round-half-to-even，并在加 `zero_point_C` 后截断到 `[-128,127]`。

二维 Grid 的 x/y 维分别覆盖 `N` 列与 `M` 行，不依赖示例或题面性能尺寸。

## 参考实现

```cpp
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

__global__ void int8_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C,
                                   int M, int N, int K, float scale_A, float scale_B,
                                   float scale_C, int zero_point_A, int zero_point_B,
                                   int zero_point_C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k) {
            int a = static_cast<int>(A[row * K + k]) - zero_point_A;
            int b = static_cast<int>(B[k * N + col]) - zero_point_B;
            acc += a * b;
        }

        float scaled = __fmul_rn(static_cast<float>(acc), scale_A);
        scaled = __fmul_rn(scaled, scale_B);
        scaled = __fdiv_rn(scaled, scale_C);
        float quantized = rintf(scaled) + static_cast<float>(zero_point_C);
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        C[row * N + col] = static_cast<int8_t>(static_cast<int>(quantized));
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    int8_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K, scale_A, scale_B,
                                            scale_C, zero_point_A, zero_point_B,
                                            zero_point_C);
    cudaDeviceSynchronize();
}
```
