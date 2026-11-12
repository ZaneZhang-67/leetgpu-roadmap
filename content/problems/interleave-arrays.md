---
id: interleave-arrays
leetgpuId: 63
titleEn: Interleave Arrays
titleZh: 数组交错排列
description: 使用 CUDA 将两个等长 FP32 数组交错写入长度为 2N 的输出数组
problemUrl: https://leetgpu.com/challenges/interleave-arrays
updatedAt: 2026-11-12
---
## 题目定义

给定两个长度均为 `N` 的 FP32 数组 `A` 和 `B`，生成长度为 `2N` 的输出：

```text
[A[0], B[0], A[1], B[1], ..., A[N-1], B[N-1]]
```

### 示例

```text
输入：A = [1.0, 2.0, 3.0]，B = [4.0, 5.0, 6.0]
输出：[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

输入：A = [10.0, 20.0]，B = [30.0, 40.0]
输出：[10.0, 30.0, 20.0, 40.0]
```

**实现要求与测试**

- 只使用原生功能，`solve` 签名保持不变
- 最终结果写入 `output`
- `1 ≤ N ≤ 50,000,000`
- `A`、`B` 的形状均为 `(N,)`，`output` 的形状为 `(2N,)`
- 三者类型均为 FP32
- 功能测试包含 `N=1`、正负混合、零、大数和 `N=100,000`
- 性能测试使用 `N=25,000,000`，输出含 50,000,000 个元素
- 绝对与相对误差容限均为 `1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
```

`A`、`B` 和 `output` 都在 GPU 显存中。输入长度相同，输出空间由平台按 `2N` 分配。平台已提供空的 `interleave_kernel`、每个 Block 256 个线程的启动配置和同步调用，作答时只需补全 kernel 函数体。

## 实现思路

一个线程对应输入下标 `i`，同时完成两次写入：

```text
output[2*i]   = A[i]
output[2*i+1] = B[i]
```

只需判断 `i<N`，便能保证最大写入位置 `2N-1` 合法。每个线程写入互不重叠的两个位置，无需原子操作或同步。

性能规模较大时，输入和输出访问仍是连续模式：相邻线程读取连续的 `A`、`B`，并写入相邻的输出对

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B,
                                  float* output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        output[2 * index] = A[index];
        output[2 * index + 1] = B[index];
    }
}

// A, B, output are device pointers
extern "C" void solve(const float* A, const float* B,
                      float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    interleave_kernel<<<blocks, threads>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
```
