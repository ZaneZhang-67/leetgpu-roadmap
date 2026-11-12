---
id: 2d-max-pooling
leetgpuId: 42
titleEn: 2D Max Pooling
titleZh: 二维最大池化
description: 在 NCHW 张量上实现带步幅和填充的二维最大池化，推导输出形状、窗口坐标与边界处理
problemUrl: https://leetgpu.com/challenges/2d-max-pooling
updatedAt: 2026-11-12
---
## 题目定义

输入按 `NCHW` 布局存储，形状为 `[N, C, H, W]`对每个 batch、通道和输出位置，在对应的二维窗口中取最大值

输出空间尺寸为：

```text
H_out = floor((H + 2*padding - kernel_size) / stride) + 1
W_out = floor((W + 2*padding - kernel_size) / stride) + 1
```

输入和输出都按行主序展平

### 实现要求与评测规模

- `solve` 函数签名必须保持不变
- 不允许使用外部库
- 结果写入 `output`
- kernel、stride、padding 均由参数给出
- `1 ≤ N ≤ 100`，`1 ≤ C ≤ 512`
- `1 ≤ H, W ≤ 1024`
- `1 ≤ kernel_size, stride ≤ 16`，`0 ≤ padding ≤ 16`
- 输入与输出元素均为 32 位浮点数
- 性能测试使用 `N=4`、`C=64`、`H=W=256`
- 性能测试参数为 `kernel_size=3`、`stride=2`、`padding=1`

### 示例

```text
input = [[1,2,3],
         [4,5,6],
         [7,8,9]]
kernel_size = 2, stride = 1, padding = 0
output = [[5,6],
          [8,9]]
```

每个输出元素对应一个 `2 × 2` 窗口

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {}
```

NCHW 输入下标为：

```text
((n * C + c) * H + h) * W + w
```

输出使用相同布局，只需把 `H`、`W` 换成 `H_out`、`W_out`

参考实现将展平后的每个输出元素分配给一个线程，再从线性下标还原 `n`、`c`、`out_h`、`out_w`

## 实现思路

### 1. 输出位置对应的输入窗口

输出位置 `(out_h, out_w)` 的窗口左上角坐标为：

```text
start_h = out_h * stride - padding
start_w = out_w * stride - padding
```

随后枚举 `kernel_size × kernel_size` 个候选位置

### 2. 填充边界

评测参考使用 PyTorch `max_pool2d`，填充区域不应引入一个可能压过负数输入的零值因此，窗口最大值从负无穷初始化，并且只读取满足下列条件的真实输入：

```text
0 ≤ in_h < H
0 ≤ in_w < W
```

这也能正确处理题库中含负无穷和大 padding 的情况

### 3. 展平线程映射

输出元素总数是 `N * C * H_out * W_out`，乘法与线性下标必须使用 `size_t`，避免合法的大张量超过 32 位整数范围线程通过 Grid-stride loop 各自负责一组输出元素，不同线程写入不同位置，无需同步

kernel 为 `1`、stride 为 `1`、padding 为 `0` 时，池化退化为原样复制；kernel 覆盖整个输入时，每个 batch 和通道只产生一个最大值

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void max_pool_2d(const float* input, float* output,
                            int N, int C, int H, int W,
                            int kernel_size, int stride, int padding,
                            int H_out, int W_out) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = (size_t)blockDim.x * gridDim.x;
    size_t total = (size_t)N * C * H_out * W_out;

    for (; index < total; index += grid_stride) {
        size_t value = index;
        int out_w = (int)(value % W_out);
        value /= W_out;
        int out_h = (int)(value % H_out);
        value /= H_out;
        int c = (int)(value % C);
        int n = (int)(value / C);

        int start_h = out_h * stride - padding;
        int start_w = out_w * stride - padding;
        float maximum = -CUDART_INF_F;

        for (int kh = 0; kh < kernel_size; ++kh) {
            int in_h = start_h + kh;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_w = start_w + kw;
                if (in_w < 0 || in_w >= W) continue;
                size_t input_index = ((size_t)(n * C + c) * H + in_h) * W + in_w;
                maximum = fmaxf(maximum, input[input_index]);
            }
        }
        output[index] = maximum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    size_t total = (size_t)N * C * H_out * W_out;
    int threads = 256;
    size_t required_blocks = (total + threads - 1) / threads;
    int blocks = required_blocks > 65535 ? 65535 : (int)required_blocks;
    max_pool_2d<<<blocks, threads>>>(input, output, N, C, H, W,
                                     kernel_size, stride, padding,
                                     H_out, W_out);
    cudaDeviceSynchronize();
}
```
