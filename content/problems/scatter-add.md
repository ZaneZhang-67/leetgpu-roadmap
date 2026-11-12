---
id: scatter-add
leetgpuId: 125
titleEn: Scatter Add
titleZh: 散射加法
description: 使用 CUDA 原子加法将源矩阵行累加到指定输出行
problemUrl: https://leetgpu.com/challenges/scatter-add
updatedAt: 2026-11-12
---
## 题目定义

给定 `src(N,D)` 和 INT32 `index(N)`，将每个源行累加到零初始化的 `output(M,D)`：

```text
output[m,d] = Σ src[i,d]，其中 index[i]=m
```

多个源行可以映射到同一输出行；没有源行映射的输出行必须为零。输入输出均按行主序存储。

### 示例

```text
src=[[1,2],[3,4],[5,6],[7,8]], index=[0,2,0,1]
output=[[6,8],[7,8],[3,4]]
```

**约束与测试**

- `1≤N≤2,000,000`，`1≤D≤1024`，`1≤M≤1,000,000`
- `0≤index[i]<M`，src 位于 `[-10,10]`
- 测试覆盖无人映射行、全零、全部映射同一行和非二次幂形状
- 性能测试 `N=1,000,000,D=128,M=50,000`，容差 `atol=rtol=1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {}
```

官方 starter 只提供空入口。output 的初始内容未定义，因此累加前必须清零全部 `M*D` 个元素。

## 实现思路

先用 `cudaMemset` 把 output 清零。随后把 `(i,d)` 展平为 `N*D` 个任务，每个线程读取 `index[i]`，对 `output[index[i],d]` 执行 `atomicAdd`。重复索引会让多个线程写同一地址，普通加法会产生数据竞争，必须使用原子操作。

无映射行保留 memset 得到的零。线性元素数可能超过 32 位乘法的安全范围，因此总量和偏移使用 `size_t`。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void scatter_kernel(const float* src, const int* index,
                               float* output, size_t total, int D) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        size_t row = i / (size_t)D;
        int col = (int)(i % (size_t)D);
        size_t dst = (size_t)index[row] * D + (size_t)col;
        atomicAdd(output + dst, src[i]);
    }
}

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {
    size_t output_count = (size_t)M * D;
    cudaMemset(output, 0, output_count * sizeof(float));
    size_t total = (size_t)N * D;
    int threads = 256;
    size_t needed = (total + threads - 1) / threads;
    int blocks = needed > 4096 ? 4096 : (int)needed;
    scatter_kernel<<<blocks, threads>>>(src, index, output, total, D);
    cudaDeviceSynchronize();
}
```
