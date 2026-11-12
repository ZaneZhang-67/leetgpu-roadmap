---
id: gaussian-error-gated-linear-unit
leetgpuId: 65
titleEn: Gaussian Error Gated Linear Unit
titleZh: 高斯误差门控线性单元
description: 使用 CUDA 实现 GEGLU 前向计算，将一维输入等分为两半，对后半段应用精确 GELU 后与前半段逐元素相乘
problemUrl: https://leetgpu.com/challenges/gaussian-error-gated-linear-unit
updatedAt: 2026-11-12
---
## 题目定义

给定长度为偶数 `N` 的 FP32 一维数组 `input`，将它从中间分成等长的 `x1` 和 `x2`：

```text
x1[i] = input[i]
x2[i] = input[N / 2 + i]
```

先对 `x2` 使用精确 GELU：

```text
GELU(x) = 0.5 × x × (1 + erf(x / √2))
```

再计算 GEGLU：

```text
output[i] = x1[i] × GELU(x2[i])
```

输出长度为 `N / 2`

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `output`

### 示例 1

```text
输入：[1.0, 1.0]，N = 2
输出：[0.8413447]
```

### 示例 2

```text
输入：[2.0, -1.0, 1.0, 0.5]，N = 4
输出：[1.6826895, -0.3457312]
```

### 约束

- `1 ≤ N ≤ 1,000,000`
- `N` 是偶数
- 输入值范围为 `[-100.0, 100.0]`
- 性能测试使用 `N = 1,000,000`
- 判定的绝对误差和相对误差均为 `1e-4`

## 模板代码解析

平台提供了空 kernel 和完整启动框架：

```cpp
#include <cuda_runtime.h>

__global__ void geglu_kernel(
    const float* input,
    float* output,
    int halfN
) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, halfN
    );
    cudaDeviceSynchronize();
}
```

- `input` 是长度为 `N` 的设备数组
- `output` 是长度为 `N / 2` 的设备数组
- starter 已计算 `halfN` 和 Grid 大小
- 作答只需补全 kernel，保持 `solve` 签名和两半连续存储的布局不变

## 实现思路

一个线程计算一个输出元素。线程索引 `i` 同时对应 `x1[i]` 和 `x2[i]`：

```cpp
float x1 = input[i];
float x2 = input[halfN + i];
```

必须使用 reference 中基于 `erf` 的精确 GELU，而不是常见的 `tanh` 近似：

```cpp
float gelu = 0.5f * x2 * (1.0f + erff(x2 * rsqrtf(2.0f)));
```

最后写入：

```cpp
output[i] = x1 * gelu;
```

边界判断使用 `i < halfN`，因此最后一个 Block 未占满时不会越界。相邻线程连续读取 `x1`、`x2` 并连续写出结果，访问模式合并良好

时间复杂度为 `O(N)`，无需临时缓冲区

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void geglu_kernel(
    const float* input,
    float* output,
    int halfN
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= halfN) return;

    float x1 = input[i];
    float x2 = input[halfN + i];
    float gelu = 0.5f * x2
        * (1.0f + erff(x2 * 0.7071067811865475f));
    output[i] = x1 * gelu;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, halfN
    );
    cudaDeviceSynchronize();
}
```
