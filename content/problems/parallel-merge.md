---
id: parallel-merge
leetgpuId: 71
titleEn: Parallel Merge
titleZh: 并行归并
description: 使用 CUDA Merge Path 将两个升序 FP32 数组稳定归并为一个升序数组
problemUrl: https://leetgpu.com/challenges/parallel-merge
updatedAt: 2026-11-12
---
## 题目定义

给定两个已经按升序排列的 FP32 数组 `A` 和 `B`，长度分别为 `M`、`N`，将全部元素归并到长度 `M+N` 的 `C` 中。结果同样升序，重复值必须全部保留。

### 示例 1

```text
A = [1.0, 3.0, 5.0, 7.0]，M = 4
B = [2.0, 4.0, 6.0, 8.0]，N = 4
C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

### 示例 2

```text
A = [-1.0, 1.0, 3.0]，M = 3
B = [2.0]，N = 1
C = [-1.0, 1.0, 2.0, 3.0]
```

约束为 `1≤M,N≤50,000,000`、`M+N≤50,000,000`；输入元素为 FP32，`A` 与 `B` 都已按非降序排列。官方测试包含长度 1、不等长数组、负数、重复值、无交错区间和非二次幂尺寸；性能用例为 `M=N=25,000,000`，要求逐元素完全一致。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const float* A, const float* B, float* C, int M, int N)
```

三个数组均为设备指针。输入已经有序，不需要先排序。相等时选择 `A` 可以给出稳定的 tie 规则；虽然平台只比较数值，这一规则也保证重复值计数正确。

## 实现思路

对输出对角线 `diag∈[0,M+N)`，寻找输入划分 `(i,j)`，满足 `i+j=diag`，且归并序列在该位置之前恰有 `i` 个 A 元素和 `j` 个 B 元素。

二分条件使用：

```text
A[i-1] ≤ B[j]
B[j-1] < A[i]
```

第二个条件采用严格小于，因此相等元素优先取自 A。确定划分后，当前位置取 `min(A[i],B[j])`，相等时仍取 A。每个线程独立计算一个输出位置，复杂度为 `O((M+N)log(min(M,N)))`。每个位置只写一次，也不需要临时缓冲。这是覆盖全部约束的正确性基线；逐输出二分并非面向最大性能用例的最优归并方案。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void merge_path_kernel(const float* A, const float* B, float* C, int M, int N) {
    int diag = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M + N;
    if (diag >= total) return;

    int low = diag - N;
    if (low < 0) low = 0;
    int high = diag;
    if (high > M) high = M;

    while (low < high) {
        int i = low + (high - low) / 2;
        int j = diag - i;
        if (i < M && j > 0 && B[j - 1] >= A[i]) low = i + 1;
        else high = i;
    }

    int i = low;
    int j = diag - i;
    if (i < M && (j >= N || A[i] <= B[j])) C[diag] = A[i];
    else C[diag] = B[j];
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    constexpr int threads = 256;
    int total = M + N;
    int blocks = (total + threads - 1) / threads;
    merge_path_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
```
