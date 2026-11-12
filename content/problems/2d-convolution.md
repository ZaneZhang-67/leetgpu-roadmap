---
id: 2d-convolution
leetgpuId: 10
titleEn: 2D Convolution
titleZh: 二维卷积
description: 使用 CUDA 实现 valid 边界的二维卷积，推导行主序索引、输出形状和线程到输出元素的映射
problemUrl: https://leetgpu.com/challenges/2d-convolution
updatedAt: 2026-11-12
---
## 题目定义

给定二维输入矩阵 `input` 和二维卷积核 `kernel`，在 GPU 上计算 valid 二维卷积valid 表示卷积核只有在完整落入输入矩阵时才参与计算，不在边缘补零

输入和卷积核都由 32 位浮点数组成，并按行主序展平为一维数组输出同样按行主序写入 `output`

输出形状为：

```text
output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1
```

题面公式为：

```text
output[i][j] = Σ Σ input[i + m][j + n] × kernel[m][n]
```

其中 `0 ≤ m < kernel_rows`，`0 ≤ n < kernel_cols`题面直接使用 `kernel[m][n]`，不额外翻转卷积核

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入数组 `output`

### 示例 1

```text
输入 input（3×3）：
[1 2 3]
[4 5 6]
[7 8 9]

kernel（2×2）：
[0 1]
[1 0]

输出 output（2×2）：
[ 6  8]
[12 14]
```

例如：

```text
output[0,0] = 1×0 + 2×1 + 4×1 + 5×0 = 6
```

### 示例 2

```text
输入 input（4×4）：
[1 1 1 1]
[1 2 3 1]
[1 4 5 1]
[1 1 1 1]

kernel（1×3）：
[1 0 1]

输出 output（4×2）：
[2 2]
[4 3]
[6 5]
[2 2]
```

### 约束

- `1 ≤ input_rows, input_cols ≤ 3072`
- `1 ≤ kernel_rows, kernel_cols ≤ 31`
- `kernel_rows ≤ input_rows`
- `kernel_cols ≤ input_cols`
- 性能测试使用 `input_rows = 3072`、`input_cols = 3072`、`kernel_rows = 15`、`kernel_cols = 15`

## 模板代码解析

CUDA starter 只提供头文件、设备指针说明和空的 `solve`：

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {}
```

`input`、`kernel`、`output` 已经是设备指针，不需要在 `solve` 中申请或复制显存需要补充 GPU kernel、启动配置和同步操作，同时保持 `solve` 的参数顺序与类型不变

参数分别表示输入矩阵的行列数和卷积核的行列数平台为 `output` 提供足够空间，结果按 `output_rows × output_cols` 的行主序布局写入

## 实现思路

### 1. 一个线程计算一个输出元素

用二维线程坐标对应输出坐标：

```cpp
int out_col = blockIdx.x * blockDim.x + threadIdx.x;
int out_row = blockIdx.y * blockDim.y + threadIdx.y;
```

合法范围为：

```cpp
out_row < output_rows && out_col < output_cols
```

Grid 两个维度均采用向上取整，因此边缘 Block 中可能存在多余线程，访问数组前必须做这项检查

### 2. 行主序索引

输入矩阵每行有 `input_cols` 个元素，因此 `input[row][col]` 的线性下标为：

```cpp
row * input_cols + col
```

卷积核每行有 `kernel_cols` 个元素：

```cpp
kernel_row * kernel_cols + kernel_col
```

输出矩阵每行有 `output_cols` 个元素：

```cpp
out_row * output_cols + out_col
```

步长必须来自对应张量的列数，不能把输入、卷积核和输出的行宽混用

### 3. 滑动窗口逐项乘加

输出坐标 `(out_row, out_col)` 同时也是当前窗口在输入中的左上角卷积核元素 `(kr, kc)` 对应的输入坐标为：

```text
(out_row + kr, out_col + kc)
```

代码为：

```cpp
float sum = 0.0f;
for (int kr = 0; kr < kernel_rows; ++kr) {
    for (int kc = 0; kc < kernel_cols; ++kc) {
        int input_idx = (out_row + kr) * input_cols + out_col + kc;
        int kernel_idx = kr * kernel_cols + kc;
        sum += input[input_idx] * kernel[kernel_idx];
    }
}
```

valid 输出形状保证最后一个合法线程的窗口仍完整位于输入内：

```text
out_row + kernel_rows - 1 ≤ input_rows - 1
out_col + kernel_cols - 1 ≤ input_cols - 1
```

循环结束后，将累加值写入当前输出位置

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
                                      int input_rows, int input_cols, int kernel_rows,
                                      int kernel_cols, int output_rows, int output_cols) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_row < output_rows && out_col < output_cols) {
        float sum = 0.0f;

        for (int kr = 0; kr < kernel_rows; ++kr) {
            for (int kc = 0; kc < kernel_cols; ++kc) {
                int input_idx = (out_row + kr) * input_cols + out_col + kc;
                int kernel_idx = kr * kernel_cols + kc;
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }

        output[out_row * output_cols + out_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((output_cols + threads_per_block.x - 1) / threads_per_block.x,
                         (output_rows + threads_per_block.y - 1) / threads_per_block.y);

    convolution_2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, output_rows,
        output_cols);
    cudaDeviceSynchronize();
}
```
