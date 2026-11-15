---
id: value-clipping
leetgpuId: 62
titleEn: Value Clipping
titleZh: 数值裁剪
description: 使用 CUDA 将一维 FP32 向量逐元素限制在闭区间 [lo,hi]
problemUrl: https://leetgpu.com/challenges/value-clipping
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `N` 的 FP32 输入向量和上下界 `lo`、`hi`，将每个元素裁剪到闭区间 `[lo,hi]`：

```text
x < lo → lo
x > hi → hi
其他   → x
```

边界值本身保持不变

### 示例

```text
输入：[1.5, -2.0, 3.0, 4.5]，lo = 0.0，hi = 3.5
输出：[1.5, 0.0, 3.0, 3.5]

输入：[-1.0, 2.0, 5.0]，lo = -0.5，hi = 2.5
输出：[-0.5, 2.0, 2.5]
```

**实现要求与测试**

- 只使用原生功能，`solve` 签名保持不变
- 最终结果写入长度为 `N` 的 `output`
- `1 ≤ N ≤ 100,000`，`lo ≤ hi`
- 输入值范围为 `[-10^6,10^6]`
- 功能测试覆盖全零、全负、正负混合、非整数上下界和 `N=2048`
- 性能测试使用 `N=100,000`、`lo=-51.24`、`hi=39.51`
- 绝对与相对误差容限均为 `1e-5`

官方 `challenge.py` 的正式测试不包含 NaN 或无穷值，因此实现以有限 FP32 输入为约束。若输入为 NaN，下面的显式比较会让它原样传播；这与常见逐比较裁剪语义一致，但不属于正式评测范围。

## 模板代码解析

```cpp
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
```

输入、输出位于 GPU 显存，上下界按值传入。平台已提供空的 `clip_kernel`、每个 Block 256 个线程的启动配置和同步调用；作答时只需补全 kernel 函数体，并保持参数顺序 `input,output,lo,hi,N`。

## 实现思路

一个线程负责一个元素，先计算全局下标并检查 `index<N`。裁剪使用两次显式比较：

```cpp
if (value < lo) value = lo;
if (value > hi) value = hi;
```

闭区间边界不会被改变；当 `lo==hi` 时，所有有限输入都会得到该值。相邻线程读写相邻位置，不需要同步。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output,
                            float lo, float hi, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        float value = input[index];
        if (value < lo) value = lo;
        if (value > hi) value = hi;
        output[index] = value;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output,
                      float lo, float hi, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    clip_kernel<<<blocks, threads>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
```
