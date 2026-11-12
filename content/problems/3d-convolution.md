---
id: 3d-convolution
leetgpuId: 11
titleEn: 3D Convolution
titleZh: 三维卷积
description: 使用 CUDA 实现 valid 边界的三维卷积，推导 depth-row-col 布局、输出形状和三维索引
problemUrl: https://leetgpu.com/challenges/3d-convolution
updatedAt: 2026-11-12
---
## 题目定义

给定三维输入体 `input` 和三维卷积核 `kernel`，计算不带 padding 的 valid 三维卷积输入、卷积核和输出都由 32 位浮点数组成，并展平为一维数组

题面将维度写成 `(depth, rows, cols)`，输出形状为：

```text
output_depth = input_depth - kernel_depth + 1
output_rows  = input_rows  - kernel_rows  + 1
output_cols  = input_cols  - kernel_cols  + 1
```

输出坐标 `(od, orow, ocol)` 的计算公式为：

```text
output(od, orow, ocol)
= Σ Σ Σ input(od + kd, orow + kr, ocol + kc) × kernel(kd, kr, kc)
```

题面直接使用卷积核当前顺序，不对三个维度做翻转

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

### 示例 1

输入体形状为 `3×3×3`：

```text
depth 0:       depth 1:       depth 2:
[1 2 3]        [10 11 12]     [19 20 21]
[4 5 6]        [13 14 15]     [22 23 24]
[7 8 9]        [16 17 18]     [25 26 27]
```

卷积核形状为 `2×3×3`：

```text
depth 0:       depth 1:
[1 0 0]        [1 1 0]
[1 1 1]        [1 1 0]
[0 0 0]        [0 0 1]
```

输出形状为 `2×1×1`：

```text
[82, 163]
```

### 示例 2

```text
input（2×2×2）：
depth 0:       depth 1:
[1 2]          [5 6]
[3 4]          [7 8]

kernel（2×2×2）：所有元素均为 1
output（1×1×1）：[36]
```

### 约束

- `1 ≤ input_depth, input_rows, input_cols ≤ 256`
- `1 ≤ kernel_depth, kernel_rows, kernel_cols ≤ 5`
- `kernel_depth ≤ input_depth`
- `kernel_rows ≤ input_rows`
- `kernel_cols ≤ input_cols`
- 题面注明性能测试使用 `input_rows = 128`、`input_cols = 128`、`kernel_rows = 5`、`kernel_cols = 5`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {}
```

平台已经把 `input`、`kernel`、`output` 放在 GPU 显存中需要在 `solve` 内补充 kernel 启动配置和同步，并保持所有参数的类型、顺序与数量不变

题面所说的“row-major, then depth”结合 `(depth, rows, cols)` 维度和示例，可按每个 depth 平面连续、平面内逐行存储理解：`cols` 是最快变化的维度，其次是 `rows`，最后是 `depth`

## 实现思路

### 1. 展平三维输出空间

输出元素总数为：

```cpp
int output_size = output_depth * output_rows * output_cols;
```

一个线程负责一个线性输出下标 `id`用一维 Grid 可以避免 CUDA Grid 三个维度的额外限制，也便于统一做边界判断：

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id >= output_size) return;
```

### 2. 从线性下标还原三维坐标

因为 `output_cols` 是最快变化的维度：

```cpp
int out_col = id % output_cols;
int tmp = id / output_cols;
int out_row = tmp % output_rows;
int out_depth = tmp / output_rows;
```

对应的线性化公式为：

```text
id = (out_depth × output_rows + out_row) × output_cols + out_col
```

### 3. 输入与卷积核索引

输入坐标 `(d, r, c)` 的下标为：

```cpp
(d * input_rows + r) * input_cols + c
```

卷积核坐标 `(kd, kr, kc)` 的下标为：

```cpp
(kd * kernel_rows + kr) * kernel_cols + kc
```

这两个公式都体现了 `cols` 最快变化、`depth` 最慢变化的布局

### 4. 三重循环累加

线程负责的输出坐标同时是当前三维窗口的起点：

```cpp
float sum = 0.0f;
for (int kd = 0; kd < kernel_depth; ++kd) {
    for (int kr = 0; kr < kernel_rows; ++kr) {
        for (int kc = 0; kc < kernel_cols; ++kc) {
            int input_idx = ((out_depth + kd) * input_rows + out_row + kr) * input_cols
                            + out_col + kc;
            int kernel_idx = (kd * kernel_rows + kr) * kernel_cols + kc;
            sum += input[input_idx] * kernel[kernel_idx];
        }
    }
}
```

valid 输出形状保证三个方向上的最后一次访问都落在输入范围内循环完成后直接写入 `output[id]`

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void convolution_3d_kernel(const float* input, const float* kernel, float* output,
                                      int input_rows, int input_cols, int kernel_depth,
                                      int kernel_rows, int kernel_cols, int output_depth,
                                      int output_rows, int output_cols) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = output_depth * output_rows * output_cols;

    if (id < output_size) {
        int out_col = id % output_cols;
        int tmp = id / output_cols;
        int out_row = tmp % output_rows;
        int out_depth = tmp / output_rows;
        float sum = 0.0f;

        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kr = 0; kr < kernel_rows; ++kr) {
                for (int kc = 0; kc < kernel_cols; ++kc) {
                    int input_idx = ((out_depth + kd) * input_rows + out_row + kr) * input_cols
                                    + out_col + kc;
                    int kernel_idx = (kd * kernel_rows + kr) * kernel_cols + kc;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }

        output[id] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    int output_size = output_depth * output_rows * output_cols;

    int threads_per_block = 256;
    int blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;

    convolution_3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input, kernel, output, input_rows, input_cols, kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols);
    cudaDeviceSynchronize();
}
```
