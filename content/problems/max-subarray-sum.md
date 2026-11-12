---
id: max-subarray-sum
leetgpuId: 51
titleEn: Max Subarray Sum
titleZh: 固定窗口最大子数组和
description: 使用 CUDA 计算长度恰好为 window_size 的连续子数组最大和
problemUrl: https://leetgpu.com/challenges/max-subarray-sum
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 INT32 数组 `input`，在所有长度恰好为 `window_size` 的连续子数组中求最大元素和，结果写入 `output[0]`。

这不是任意长度的最大子数组问题。合法窗口起点为 `0..N-window_size`，每个窗口都必须包含恰好 `window_size` 个元素。

示例：`input=[1,2,4,2,3]`、`window_size=2` 时，各窗口和为 `3,6,6,5`，输出为 `6`。全负数也必须选一个合法窗口；`[-1,-4,-2,1]`、窗口长度 3 的结果是 `-5`。

题面约束为 `1≤N≤50000`、`-10≤input[i]≤10`、`1≤window_size≤N`。官方测试包含窗口长度 1、首窗口最大、全负数和 `window_size=N`；性能用例为 `N=50000`、`window_size=25000`。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const int* input, int* output, int N, int window_size)
```

`input` 和 `output` 都是设备指针。输入输出类型均为 INT32，输出只有一个元素。`challenge.py` 的 reference 先保留首窗口的和，再逐步滑动，因此实现不能把最大值错误初始化为零，也不能漏掉起点为 0 的窗口。

## 实现思路

约束下最大绝对窗口和不超过 `50000×10=500000`，INT32 足够。参考实现用一个 GPU 线程完成滑动窗口：先累加前 `window_size` 个元素，将其作为 `current_sum` 和 `max_sum`；之后每右移一步，减去离开窗口的元素并加上新元素。

这种实现为 `O(N)`，只读取首窗口一次，后续每步读取两个元素。单线程并行度有限，但性能用例只有 5 万个整数，避免了为单个标量结果设计复杂的前缀和与归约临时缓冲。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void max_window_sum_kernel(const int* input, int* output, int N, int window_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int current_sum = 0;
    for (int i = 0; i < window_size; ++i) current_sum += input[i];
    int max_sum = current_sum;

    for (int i = window_size; i < N; ++i) {
        current_sum += input[i] - input[i - window_size];
        if (current_sum > max_sum) max_sum = current_sum;
    }
    output[0] = max_sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    max_window_sum_kernel<<<1, 1>>>(input, output, N, window_size);
    cudaDeviceSynchronize();
}
```
