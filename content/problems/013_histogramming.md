---
id: histogramming
leetgpuId: 13
titleEn: Histogramming
titleZh: 直方图统计
description: 使用 CUDA 原子操作统计整数数组中各个取值的出现次数，并说明并发写冲突与输出初始化
problemUrl: https://leetgpu.com/challenges/histogramming
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 32 位整数数组 `input` 和桶数量 `num_bins`，统计取值范围 `[0, num_bins)` 中每个整数出现的次数

输出数组 `histogram` 长度为 `num_bins`，其中：

```text
histogram[b] = input 中值等于 b 的元素个数
```

题面保证每个输入值都满足：

```text
0 ≤ input[i] < num_bins
```

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入数组 `histogram`

### 示例 1

```text
输入：input = [0, 1, 2, 1, 0]，N = 5，num_bins = 3
输出：[2, 2, 1]
```

`0` 出现两次，`1` 出现两次，`2` 出现一次

### 示例 2

```text
输入：input = [3, 3, 3, 3]，N = 4，num_bins = 5
输出：[0, 0, 0, 4, 0]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- `0 ≤ input[i] < num_bins`
- `1 ≤ num_bins ≤ 1024`
- 性能测试使用 `N = 50,000,000`、`num_bins = 256`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {}
```

`input` 和 `histogram` 都是平台提供的设备指针`input` 只读，`histogram` 保存最终计数需要在 `solve` 中完成输出清零、统计 kernel 启动和同步，同时保持函数签名不变

直方图计数依赖初始值为零，不能依赖输出缓冲区原有内容因此参考实现会显式调用：

```cpp
cudaMemset(histogram, 0, num_bins * sizeof(int));
```

## 实现思路

### 1. Grid-stride loop 覆盖全部输入

`N` 最大为一亿一个线程可以通过固定步长处理多个元素：

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = id; i < N; i += stride) {
    // 处理 input[i]
}
```

这样启动规模不必随 `N` 无限增大，同时所有输入下标仍会被覆盖一次

### 2. 为什么普通加法会出错

不同输入位置可能具有相同值例如多个线程同时遇到值 `3`，都会更新 `histogram[3]`

普通表达式：

```cpp
histogram[bin] += 1;
```

会经历读取、加一、写回三个步骤如果多个线程同时读取旧值，其中一些更新会互相覆盖，最终计数偏小

### 3. 使用原子加法

CUDA 的 `atomicAdd` 会让同一个地址上的加法原子完成：

```cpp
int bin = input[i];
if (bin >= 0 && bin < num_bins) {
    atomicAdd(&histogram[bin], 1);
}
```

题面约束给出的输入都位于 `[0, num_bins)`；这里仍保留范围判断，使地址计算与平台参考语义一致，遇到越界值时直接忽略，避免访问直方图外部内存

这个朴素实现直接在全局直方图上执行原子操作当数据集中在少数桶时，多个线程会竞争相同地址，但实现逻辑简单，能够作为正确性基线

### 4. 启动配置

参考实现使用每个 Block 256 个线程，并限制 Block 数：

```cpp
int threads_per_block = 256;
int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
if (blocks_per_grid > 4096) blocks_per_grid = 4096;
```

Grid-stride loop 保证即使 Block 数被限制，所有 `N` 个元素仍能处理限制启动规模也避免为超大输入创建不必要的 Block

### 5. 可继续优化的方向

`num_bins ≤ 1024`，可以让每个 Block 先在 Shared Memory 中构建局部直方图，再把局部结果合并到全局数组这样可减少全局原子操作次数，但需要额外处理 Shared Memory 初始化、同步和最终合并

本题参考实现采用全局 `atomicAdd`，重点是正确处理并发写冲突和输出初始化

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void histogram_kernel(
    const int* input,
    int* histogram,
    int N,
    int num_bins
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&histogram[bin], 1);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 4096) {
        blocks_per_grid = 4096;
    }

    histogram_kernel<<<blocks_per_grid, threads_per_block>>>(
        input, histogram, N, num_bins
    );
    cudaDeviceSynchronize();
}
```
