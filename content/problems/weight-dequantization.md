---
id: weight-dequantization
leetgpuId: 64
titleEn: Weight Dequantization
titleZh: 权重反量化
description: 使用 CUDA 按二维 tile 的共享缩放因子对 FP32 权重矩阵逐元素反量化
problemUrl: https://leetgpu.com/challenges/weight-dequantization
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `(M,N)` 的 FP32 权重矩阵 `X` 和缩放矩阵 `S`，按 `TILE_SIZE × TILE_SIZE` 的二维分组计算：

```text
Y[i,j] = X[i,j] × S[i/TILE_SIZE, j/TILE_SIZE]
```

缩放矩阵形状为：

```text
ceil(M/TILE_SIZE) × ceil(N/TILE_SIZE)
```

边缘不足一个完整 tile 的行列仍使用对应的最后一个缩放因子

### 示例

```text
M = 4，N = 4，TILE_SIZE = 2
X = [[10, 10, 5, 5],
     [10, 10, 5, 5],
     [ 2,  2, 8, 8],
     [ 2,  2, 8, 8]]
S = [[0.5, 2.0],
     [4.0, 0.25]]
Y = [[5.0, 5.0, 10.0, 10.0],
     [5.0, 5.0, 10.0, 10.0],
     [8.0, 8.0,  2.0,  2.0],
     [8.0, 8.0,  2.0,  2.0]]
```

四个 `2×2` tile 分别乘以 `S` 中的 `0.5`、`2.0`、`4.0` 和 `0.25`。；这里沿用题面示意的 `TILE_SIZE=2` 便于展示；正式评测只使用 `{16,32,64,128}`

**实现要求与测试**

- 不允许使用外部库，`solve` 签名保持不变
- 最终结果写入形状为 `(M,N)` 的 `Y`
- `1 ≤ M,N ≤ 8192`
- `TILE_SIZE ∈ {16,32,64,128}`
- 功能测试包含 `1×1`、非整除尺寸、全零输入、负数和 `2048×4096`
- 性能测试使用 `8192×8192`、`TILE_SIZE=128`
- 绝对与相对误差容限均为 `1e-5`

尽管题名使用“反量化”，官方运行时中的 `X`、`S`、`Y` 都是 FP32，公式只有乘法，没有 zero-point，也没有整数解码

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y,
                      int M, int N, int TILE_SIZE) {}
```

三个数组按行主序存储。`S` 的列数不是 `N/TILE_SIZE` 的截断结果，而是 `ceil(N/TILE_SIZE)`。模板没有提供 kernel 和启动配置。

## 实现思路

每个线程处理 `X` 的一个元素，并由线性下标计算：

```text
row = index / N
col = index % N
scaleRow = row / TILE_SIZE
scaleCol = col / TILE_SIZE
```

缩放矩阵的行主序下标为：

```text
scaleRow × ceil(N/TILE_SIZE) + scaleCol
```

只需对真实的 `M×N` 元素启动线程，因此边缘 tile 不需要额外 padding 或边界分支。`M×N` 最大约 6700 万，使用 `size_t` 计算总量和线性下标。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void dequantize_kernel(const float* X, const float* S,
                                  float* Y, size_t total, int N,
                                  int scale_cols, int tile_size) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (; index < total; index += stride) {
        int row = (int)(index / N);
        int col = (int)(index % N);
        int scale_row = row / tile_size;
        int scale_col = col / tile_size;
        float scale = S[scale_row * scale_cols + scale_col];
        Y[index] = X[index] * scale;
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y,
                      int M, int N, int TILE_SIZE) {
    size_t total = (size_t)M * N;
    int scale_cols = (N + TILE_SIZE - 1) / TILE_SIZE;
    int threads = 256;
    size_t needed = (total + threads - 1) / threads;
    int blocks = needed > 4096 ? 4096 : (int)needed;
    dequantize_kernel<<<blocks, threads>>>(X, S, Y, total, N,
                                           scale_cols, TILE_SIZE);
    cudaDeviceSynchronize();
}
```
