---
id: gaussian-blur
leetgpuId: 28
titleEn: Gaussian Blur
titleZh: 高斯模糊
description: 使用 CUDA 对行主序二维图像执行零填充卷积，理解二维索引、核中心和边界处理
problemUrl: https://leetgpu.com/challenges/gaussian-blur
updatedAt: 2026-11-12
---
## 题目定义

对二维浮点图像应用高斯模糊给定输入图像和卷积核，每个输出像素等于邻域像素与核权重的加权和输入、卷积核和输出均按行主序存储

图像边界之外的值按零处理，输出形状与输入形状相同

**实现要求**

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`
- 边界使用零填充

**示例**

```text
输入图像：5 × 5
卷积核：3 × 3
输出图像：5 × 5
```

卷积核中心覆盖输出位置 `(row, col)`，核内每个位置 `(kr, kc)` 对应输入位置：

```text
in_row = row + kr - kernel_rows / 2
in_col = col + kc - kernel_cols / 2
```

若输入坐标越界，该项贡献为零

**约束**

- `1 ≤ input_rows, input_cols ≤ 4096`
- 题面规定 `3 ≤ kernel_rows, kernel_cols ≤ 21`，且二者均为奇数；当前 `challenge.py` 另含 `1 × 1` 边界测试
- 题面规定卷积核权重非负且总和为 `1.0`；当前随机功能与性能测试直接使用未归一化权重，计算时应原样读取
- 性能测试使用 `512 × 512` 图像和 `7 × 7` 卷积核
- 绝对与相对误差容限均为 `1e-5`

## 模板代码解析

平台 CUDA starter 只提供 `solve` 签名：

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {}
```

`input` 长度为 `input_rows × input_cols`，`kernel` 长度为 `kernel_rows × kernel_cols`，`output` 与输入等长三者均位于 GPU 显存需要自行定义 kernel、配置二维 Grid 并启动计算

## 实现思路

每个线程负责一个输出像素二维线程坐标为：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

合法线程满足 `row < input_rows && col < input_cols`

核中心偏移分别是 `kernel_rows / 2` 与 `kernel_cols / 2`遍历核时先计算对应输入坐标，只有坐标位于图像内才累加：

```cpp
sum += input[in_row * input_cols + in_col]
     * kernel[kr * kernel_cols + kc];
```

越界位置不累加，等价于零填充最后把结果写入 `output[row * input_cols + col]`

`challenge.py` 使用 `torch.nn.functional.conv2d` 和 `(kernel_rows/2, kernel_cols/2)` padding 作为参考，这里的运算是互相关：卷积核既不翻转，也不重新归一化评测还覆盖 `1 × 1` 图像、`1 × 1` 核、`3 × 3` 与 `5 × 5` 核，因此实现不能假设固定核尺寸

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(const float* input, const float* kernel, float* output,
                                     int input_rows, int input_cols,
                                     int kernel_rows, int kernel_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < input_rows && col < input_cols) {
        int radius_row = kernel_rows / 2;
        int radius_col = kernel_cols / 2;
        float sum = 0.0f;

        for (int kr = 0; kr < kernel_rows; ++kr) {
            int in_row = row + kr - radius_row;
            if (in_row < 0 || in_row >= input_rows) continue;

            for (int kc = 0; kc < kernel_cols; ++kc) {
                int in_col = col + kc - radius_col;
                if (in_col < 0 || in_col >= input_cols) continue;

                sum += input[in_row * input_cols + in_col]
                     * kernel[kr * kernel_cols + kc];
            }
        }

        output[row * input_cols + col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threads(16, 16);
    dim3 blocks((input_cols + threads.x - 1) / threads.x,
                (input_rows + threads.y - 1) / threads.y);
    gaussian_blur_kernel<<<blocks, threads>>>(input, kernel, output, input_rows, input_cols,
                                              kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
```
