---
id: 1d-convolution
leetgpuId: 9
titleEn: 1D Convolution
titleZh: 一维卷积
description: 使用原生 CUDA 实现 valid 边界的一维卷积，介绍输出长度、滑动窗口、线程映射、边界判断和逐项乘加
problemUrl: https://leetgpu.com/challenges/1d-convolution
updatedAt: 2026-11-12
---
## 题目定义

编写一个 GPU 程序执行一维卷积给定输入数组 `input` 和卷积核 `kernel`，计算 valid 边界下的输出数组 `output`

valid 表示只有当卷积核完整覆盖输入数组时才计算输出，不在输入两端补零

输入包含两个一维数组：

- `input`：由 32 位浮点数组成的输入数组
- `kernel`：由 32 位浮点数组成的卷积核

输出数组长度为：

```text
output_size = input_size - kernel_size + 1
```

题目定义的计算公式为：

```text
output[i] = Σ input[i + j] × kernel[j]
```

其中：

```text
0 ≤ j < kernel_size
0 ≤ i < output_size
```

这里按题目给出的顺序直接使用 `kernel[j]`，不额外反转卷积核

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入数组 `output`

### 示例 1

```text
输入：input = [1, 2, 3, 4, 5]，kernel = [1, 0, -1]
输出：[-2, -2, -2]
```

第一个输出元素的计算过程：

```text
output[0]
= input[0] × kernel[0]
+ input[1] × kernel[1]
+ input[2] × kernel[2]
= 1 × 1 + 2 × 0 + 3 × (-1)
= -2
```

卷积核向右移动后：

```text
output[1] = 2 × 1 + 3 × 0 + 4 × (-1) = -2
output[2] = 3 × 1 + 4 × 0 + 5 × (-1) = -2
```

### 示例 2

```text
输入：input = [2, 4, 6, 8]，kernel = [0.5, 0.2]
输出：[1.8, 3.2, 4.6]
```

计算过程：

```text
output[0] = 2 × 0.5 + 4 × 0.2 = 1.8
output[1] = 4 × 0.5 + 6 × 0.2 = 3.2
output[2] = 6 × 0.5 + 8 × 0.2 = 4.6
```

### 约束

- `1 ≤ input_size ≤ 1,500,000`
- `1 ≤ kernel_size ≤ 2047`
- `kernel_size ≤ input_size`
- 性能测试使用 `input_size = 1,500,000`、`kernel_size = 2047`

## 模板代码解析

模板提供了空的 GPU kernel 和 CPU 端启动框架，需要补全 `convolution_1d_kernel` 的函数体：

```cpp
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size
) {}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size
) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, kernel, output, input_size, kernel_size
    );
    cudaDeviceSynchronize();
}
```

参数含义：

- `input`：输入数组，长度为 `input_size`
- `kernel`：卷积核，长度为 `kernel_size`
- `output`：输出数组，长度为 `output_size`
- `input_size`：输入数组长度
- `kernel_size`：卷积核长度

`input`、`kernel` 和 `output` 都指向 GPU 显存，`input` 与 `kernel` 带有 `const`，kernel 只读取它们

CPU 端先计算 valid 卷积的输出长度，再按输出元素数量启动线程每个 Block 有 256 个线程，Block 数通过向上取整计算：

```cpp
int blocksPerGrid =
    (output_size + threadsPerBlock - 1) / threadsPerBlock;
```

## 实现思路

每个输出元素都对应输入数组上的一个滑动窗口窗口长度等于 `kernel_size`，窗口起点等于输出下标 `id`

因此，可以让一个 GPU 线程负责一个输出元素：

```text
线程 0 → 计算 output[0]
线程 1 → 计算 output[1]
线程 2 → 计算 output[2]
...
线程 id → 计算 output[id]
```

### 1. 计算输出下标

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

这个 `id` 既是线程负责的输出下标，也是当前滑动窗口在输入数组中的起点

### 2. 为什么输出长度要加 1

卷积核必须完整落在输入数组内对于长度为 `input_size` 的输入和长度为 `kernel_size` 的卷积核，窗口起点最大为：

```text
input_size - kernel_size
```

起点从 `0` 开始，所以合法起点数量为：

```text
(input_size - kernel_size) - 0 + 1
= input_size - kernel_size + 1
```

例如，输入长度为 5、卷积核长度为 3 时，合法窗口为：

```text
id = 0 → input[0..2]
id = 1 → input[1..3]
id = 2 → input[2..4]
```

一共有 `5 - 3 + 1 = 3` 个输出元素

### 3. 检查输出边界

Grid 使用向上取整，最后一个 Block 可能包含没有对应输出元素的线程访问数组前需要判断：

```cpp
if (id < output_size)
```

也可以直接写成：

```cpp
if (id < input_size - kernel_size + 1)
```

`solve` 中已经计算了 `output_size`，在 kernel 内再次使用相同公式即可保证最后一个输入下标不超过 `input_size - 1`

### 4. 沿卷积核逐项乘加

线程 `id` 负责的窗口从 `input[id]` 开始对于卷积核下标 `j`，需要读取：

```cpp
input[id + j]
kernel[j]
```

将对应位置相乘并累加：

```cpp
float sum = 0.0f;

for (int j = 0; j < kernel_size; ++j) {
    sum += input[id + j] * kernel[j];
}
```

循环结束后写入：

```cpp
output[id] = sum;
```

### 5. 为什么 `input[id + j]` 不会越界

合法线程满足：

```text
id ≤ input_size - kernel_size
```

循环中的最大 `j` 为：

```text
kernel_size - 1
```

因此线程访问的最大输入下标为：

```text
id + j
≤ input_size - kernel_size + kernel_size - 1
= input_size - 1
```

这正好是输入数组最后一个合法下标

### 6. 访存特点

相邻线程负责相邻的输出元素，因此它们读取的输入窗口高度重叠例如线程 0 读取 `input[0..2]`，线程 1 读取 `input[1..3]`

朴素实现会从全局内存重复读取这些重叠数据，但索引关系清楚，适合作为正确性基线后续可以使用 Shared Memory 缓存一个 Block 所需的输入片段，也可以在适合的条件下缓存卷积核，减少重复访存

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (id < output_size) {
        float sum = 0.0f;

        for (int j = 0; j < kernel_size; ++j) {
            sum += input[id + j] * kernel[j];
        }

        output[id] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size
) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, kernel, output, input_size, kernel_size
    );
    cudaDeviceSynchronize();
}
```
