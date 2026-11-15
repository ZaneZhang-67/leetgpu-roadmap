---
id: linear-recurrence
leetgpuId: 82
titleEn: Linear Recurrence
titleZh: 线性递推
description: 使用 CUDA 按时间顺序计算批量线性递推，说明序列内依赖和批次间并行
problemUrl: https://leetgpu.com/challenges/linear-recurrence
updatedAt: 2026-11-12
---
## 题目定义

给定两个形状均为 `[B, L]` 的 FP32 行主序矩阵 `a` 和 `x`，计算同形状输出 `h`：

```text
h[b,0] = x[b,0]
h[b,t] = a[b,t] × h[b,t-1] + x[b,t]，t ≥ 1
```

同一序列的当前位置依赖前一位置，因此必须保持时间顺序；不同批次互不依赖。

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果写入 `h`

### 示例

```text
B = 2，L = 4
a = [[0.5, 0.5, 0.5, 0.5],
     [1.0, 1.0, 1.0, 1.0]]
x = [[1.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0]]
h = [[1.0, 0.5, 0.25, 0.125],
     [1.0, 2.0, 3.0, 4.0]]
```

### 约束

- `1 ≤ B ≤ 256`
- `1 ≤ L ≤ 65,536`
- `a`、`x`、`h` 均为 FP32
- 功能测试覆盖 `L = 1`、全零、`a = 0`、`a = 1` 及非 2 的幂长度
- 性能测试使用 `B = 64`、`L = 16,384`
- 绝对误差和相对误差均为 `1e-5`

## 模板代码解析

CUDA starter 只提供固定入口：

```cpp
#include <cuda_runtime.h>

// a, x, h are device pointers
extern "C" void solve(
    const float* a,
    const float* x,
    float* h,
    int B,
    int L
) {}
```

三个指针都指向设备内存，布局为 `[B, L]` 行主序。starter 没有提供 kernel 或启动配置。

## 实现思路

让一个线程负责一个批次，可以直接匹配 reference 的运算顺序。线程先将 `x[b,0]` 写入 `h[b,0]`，再让 `t` 从 1 递增到 `L-1`：

```text
state = a[b,t] × state + x[b,t]
h[b,t] = state
```

同一序列的多个时间步并不是彼此独立的普通逐元素计算；若并行处理，线程可能在前驱状态产生之前就读取它。批次维可以并行，序列维则必须在各线程内严格按顺序执行。runtime 分别执行 FP32 乘法和加法，因此代码使用 `__fmul_rn` 和 `__fadd_rn`，防止编译器将两步收缩为一次 FMA，从而改变舍入结果。

这份基线实现的时间复杂度为 `O(BL)`，额外空间为 `O(1)`。输入和输出均按连续行访问。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void linear_recurrence_kernel(
    const float* a,
    const float* x,
    float* h,
    int B,
    int L
) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= B) return;

    int offset = batch * L;
    float state = x[offset];
    h[offset] = state;

    for (int t = 1; t < L; ++t) {
        int index = offset + t;
        float product = __fmul_rn(a[index], state);
        state = __fadd_rn(product, x[index]);
        h[index] = state;
    }
}

// a, x, h are device pointers
extern "C" void solve(
    const float* a,
    const float* x,
    float* h,
    int B,
    int L
) {
    constexpr int threads = 256;
    int blocks = (B + threads - 1) / threads;
    linear_recurrence_kernel<<<blocks, threads>>>(a, x, h, B, L);
    cudaDeviceSynchronize();
}
```
