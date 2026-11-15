---
id: rgb-to-grayscale
leetgpuId: 66
titleEn: RGB to Grayscale
titleZh: RGB 转灰度图
description: 使用 CUDA 将交错存储的 RGB 浮点图像转换为灰度图，介绍像素线程映射、通道寻址和精确加权公式
problemUrl: https://leetgpu.com/challenges/rgb-to-grayscale
updatedAt: 2026-11-12
---
## 题目定义

给定一张宽为 `width`、高为 `height` 的 RGB 图像。输入使用一维 FP32 数组存储，每个像素的三个通道连续排列：

```text
R, G, B, R, G, B, ...
```

对每个像素计算：

```text
gray = 0.299 × R + 0.587 × G + 0.114 × B
```

输入长度为 `height × width × 3`，输出长度为 `height × width`

### 实现要求

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 必须使用精确系数 `0.299`、`0.587`、`0.114`
- 最终结果必须写入 `output`

### 示例 1

```text
input = [255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]
width = 2，height = 2
output = [76.245, 149.685, 29.07, 128.0]
```

### 示例 2

```text
input = [100, 150, 200]
width = 1，height = 1
output = [140.75]
```

### 约束

- `1 ≤ width, height ≤ 4096`
- `width × height ≤ 4,194,304`
- RGB 值范围为 `[0.0, 255.0]`
- 功能测试包含单像素、全零、全 255、正方形和非正方形图像
- 性能测试使用 `width = height = 2048`
- 判定的绝对误差和相对误差均为 `1e-5`

## 模板代码解析

平台提供了空 kernel 和完整启动框架：

```cpp
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(
    const float* input,
    float* output,
    int width,
    int height
) {}

// input, output are device pointers
extern "C" void solve(
    const float* input,
    float* output,
    int width,
    int height
) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, width, height
    );
    cudaDeviceSynchronize();
}
```

starter 已按像素数量计算 Grid，每个线程应负责一个输出像素。`width` 和 `height` 决定有效像素总数，但无需将一维线程索引拆成行列坐标

## 实现思路

线程的全局索引就是像素索引：

```cpp
int pixel = blockIdx.x * blockDim.x + threadIdx.x;
```

像素 `pixel` 在输入中的起点为 `3 × pixel`：

```text
R = input[3 × pixel]
G = input[3 × pixel + 1]
B = input[3 × pixel + 2]
```

输出中每个像素只占一个位置：

```cpp
output[pixel] = 0.299f * R + 0.587f * G + 0.114f * B;
```

边界条件为 `pixel < width * height`，因此支持任意宽高，包括非正方形图像

每个输入通道只读取一次，每个输出只写一次，时间复杂度为 `O(width × height)`，额外空间为 `O(1)`

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(
    const float* input,
    float* output,
    int width,
    int height
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    if (pixel >= totalPixels) return;

    int base = pixel * 3;
    float red = input[base];
    float green = input[base + 1];
    float blue = input[base + 2];

    output[pixel] =
        0.299f * red + 0.587f * green + 0.114f * blue;
}

// input, output are device pointers
extern "C" void solve(
    const float* input,
    float* output,
    int width,
    int height
) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, width, height
    );
    cudaDeviceSynchronize();
}
```
