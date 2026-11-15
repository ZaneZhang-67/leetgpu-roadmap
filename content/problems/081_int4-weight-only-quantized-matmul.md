---
id: int4-weight-only-quantized-matmul
leetgpuId: 81
titleEn: INT4 Weight-Only Quantized MatMul
titleZh: INT4 仅权重量化矩阵乘法
description: 使用 CUDA 解包带偏移的 INT4 权重，按组反量化并完成 W4A16 矩阵乘法
problemUrl: https://leetgpu.com/challenges/int4-weight-only-quantized-matmul
updatedAt: 2026-11-12
---
## 题目定义

给定 FP16 激活 `x(M,K)`、打包权重 `w_q(N,K/2)` 和 FP16 缩放 `scales(N,K/group_size)`，计算 `y=xWᵀ`。输出 `y(M,N)` 的数据类型为 FP16。

每个字节存放两个权重：高四位对应偶数 `k`，低四位对应奇数 `k`。nibble 采用无符号表示，取值范围为 `[0,15]`；真实 INT4 值为 `nibble-8`，范围为 `[-8,7]`。本题没有额外的 zero-point 数组，偏移固定为 8：

```text
W[n,k] = (nibble[n,k]-8) × scales[n,k/group_size]
```

### 示例

```text
M=2，N=4，K=4，group_size=2
x=[[1,0,1,0],[0,1,0,1]]
w_q=[[0x99,0x99],[0xAA,0xAA],[0x77,0x77],[0x88,0x88]]
scales 每项均为0.5
y=[[1,2,-1,0],[1,2,-1,0]]
```

约束为 `1≤M,N,K≤8192`，`K` 能同时被 2 和 `group_size` 整除，`group_size∈{2,4,8,16,32,64,128}`。所有张量均按行主序保存。性能用例为 `M=N=K=4096`、`group_size=128`，容差为 `1e-2`。

## 模板代码解析

starter 使用半精度数据类型，函数签名如下：

```text
extern "C" void solve(const half* x, const unsigned char* w_q, const half* scales,
                      half* y, int M, int N, int K, int group_size)
```

在 `w_q[n*(K/2)+k/2]` 中，偶数 `k` 读取高 nibble，奇数 `k` 读取低 nibble。权重布局为 `(N,K)`，因此每个输出都要与权重的第 `n` 行做点积，而不是按 `(K,N)` 布局读取。

## 实现思路

实现采用 `16×16` 的 Shared Memory tiled GEMM。每个 Block 计算 `y` 的一个 `16×16` 子块，每个线程负责其中一个输出。K 维按 16 分块：线程 `(ty,tx)` 把 `x[row,k0+tx]` 转成 FP32 后载入 `X[ty][tx]`，同时将 `W[col,k0+ty]` 解包，减去固定偏移 8，再乘以对应的 group scale，最后载入 `Wtile[ty][tx]`。

同步后，每个输出线程累加 `X[ty][i]×Wtile[i][tx]`。同一个激活值和反量化权重会被 Block 内的 16 个输出复用，从而避免每个输出都从全局内存重复读取和解包。当 K 小于 16，或最后一个 K tile 不完整时，越界位置写入零。M、N 边缘线程仍参与同步，只在最终写回时检查边界。所有乘加都采用 FP32，以匹配 reference 的 `x.float() @ W.float().T`，最后再转换为 FP16。

实现不会物化完整的反量化权重，只使用两个 `16×16` 的 FP32 Shared Memory tile，可直接作为 tiled 性能基线提交。后续还可使用向量化解包、双缓冲和 Tensor Core 进一步优化。

## 参考实现

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void int4_matmul_kernel(const half* x, const unsigned char* w_q,
                                   const half* scales, half* y,
                                   int M, int N, int K, int group_size) {
    __shared__ float X[16][16];
    __shared__ float Wtile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int n = blockIdx.x * 16 + tx;
    int m = blockIdx.y * 16 + ty;
    int packed_stride = K / 2;
    int scale_stride = K / group_size;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        int x_k = k0 + tx;
        X[ty][tx] = (m < M && x_k < K) ? __half2float(x[m * K + x_k]) : 0.0f;

        int w_k = k0 + ty;
        float weight = 0.0f;
        if (n < N && w_k < K) {
            unsigned char packed = w_q[n * packed_stride + w_k / 2];
            int nibble = (w_k & 1) == 0 ? ((packed >> 4) & 0xF) : (packed & 0xF);
            float scale = __half2float(scales[n * scale_stride + w_k / group_size]);
            weight = static_cast<float>(nibble - 8) * scale;
        }
        Wtile[ty][tx] = weight;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += X[ty][i] * Wtile[i][tx];
        __syncthreads();
    }
    if (m < M && n < N) y[m * N + n] = __float2half(sum);
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const half* x, const unsigned char* w_q, const half* scales,
                      half* y, int M, int N, int K, int group_size) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    int4_matmul_kernel<<<blocks, threads>>>(x, w_q, scales, y, M, N, K, group_size);
    cudaDeviceSynchronize();
}
```
