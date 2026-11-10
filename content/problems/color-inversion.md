---
id: color-inversion
titleEn: Color Inversion
titleZh: 图像颜色反转
description: 使用原生 CUDA 原地反转 RGBA 图像的 RGB 分量，介绍一维 RGBA 内存布局、通道判断、边界检查和模板中 Block 数少乘 4 的问题
problemUrl: https://leetgpu.com/challenges/color-inversion
updatedAt: 2026-11-10
---
## 题目定义

编写一个 GPU 程序，反转一张图像的颜色图像以一维 RGBA 数组表示，每个分量都是 8 位无符号整数 `unsigned char`

每个像素依次包含四个分量：

```text
R, G, B, A
```

颜色反转对 R、G、B 分量执行：

```text
反转值 = 255 - 原值
```

Alpha 分量保持不变结果原地写回数组 `image`

数组一共有 `width × height × 4` 个元素前四个元素表示左上角像素的 RGBA 值，接下来的四个元素表示它右侧像素的 RGBA 值，依次排列

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写回数组 `image`

### 示例 1

```text
输入：image = [255, 0, 128, 255, 0, 255, 0, 255]，width = 1，height = 2
输出：[0, 255, 127, 255, 255, 0, 255, 255]
```

两个像素分别为：

```text
[255, 0, 128, 255] → [0, 255, 127, 255]
[0, 255, 0, 255]   → [255, 0, 255, 255]
```

每个像素的第四个分量 Alpha 都保持 `255`

### 示例 2

```text
输入：image = [10, 20, 30, 255, 100, 150, 200, 255]，width = 2，height = 1
输出：[245, 235, 225, 255, 155, 105, 55, 255]
```

### 约束

- `1 ≤ width ≤ 4096`
- `1 ≤ height ≤ 4096`
- `width × height ≤ 8,388,608`
- 性能测试使用 `height = 5120`、`width = 4096`

## 模板代码解析

模板提供了空的 GPU kernel 和 CPU 端入口函数：

```cpp
#include <cuda_runtime.h>

__global__ void invert_kernel(
    unsigned char* image,
    int width,
    int height
) {}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(
    unsigned char* image,
    int width,
    int height
) {
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```

- `image` 指向 GPU 显存，输入和输出共用这段内存
- `width`、`height` 表示图像宽度和高度
- `threadsPerBlock = 256` 表示每个 Block 启动 256 个线程
- `invert_kernel<<<blocksPerGrid, threadsPerBlock>>>` 启动 kernel
- `cudaDeviceSynchronize()` 等待 GPU 执行完成

### 模板中的 Block 数少乘了 4

模板初始化 `blocksPerGrid` 时只按像素数量计算：

```cpp
int blocksPerGrid =
    (width * height + threadsPerBlock - 1) / threadsPerBlock;
```

如果 kernel 采用“一个线程处理一个 RGBA 分量”的映射，那么需要处理的元素数量是：

```text
width × height × 4
```

因此这里少乘了 `4`，正确的 Block 数应为：

```cpp
int blocksPerGrid =
    (width * height * 4 + threadsPerBlock - 1) / threadsPerBlock;
```

例如，图像只有一个像素时，数组仍有四个元素若线程下标直接对应数组下标，Grid 就必须覆盖这四个分量，而不是只覆盖一个像素

另一种写法是保留模板的 `width × height`，改为一个线程处理一个完整像素，再由该线程访问 `base + 0`、`base + 1`、`base + 2` 三个颜色分量本题解按用户代码采用第一种方式，因此修正启动配置，让一个线程对应一个分量

## 实现思路

### 1. 一个线程对应一个 RGBA 分量

线程先计算自己在整个 Grid 中的一维下标：

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

数组总长度为：

```cpp
int total = width * height * 4;
```

由于 Block 数向上取整，最后一个 Block 可能包含多余线程，因此访问数组前必须判断：

```cpp
if (id < total)
```

### 2. 判断当前分量是否为 Alpha

RGBA 每四个元素组成一个像素：

```text
下标：0  1  2  3 | 4  5  6  7 | ...
通道：R  G  B  A | R  G  B  A | ...
```

Alpha 通道出现在每组的第 4 个位置，因此可以判断：

```cpp
(id + 1) % 4 != 0
```

满足这个条件时，当前分量属于 R、G 或 B，需要执行反转；条件不满足时，当前分量是 Alpha，保持原值

等价写法是：

```cpp
id % 4 != 3
```

### 3. 原地反转 RGB

每个颜色分量都是 `unsigned char`，取值范围为 `0～255`反转公式为：

```cpp
image[id] = 255 - image[id];
```

示例映射：

```text
0   → 255
10  → 245
128 → 127
255 → 0
```

每个线程只修改一个独立数组元素，线程之间没有写冲突，也不需要在 kernel 内同步

### 4. 完整判断顺序

```cpp
if (id < width * height * 4) {
    if ((id + 1) % 4 != 0) {
        image[id] = 255 - image[id];
    }
}
```

外层判断保护数组边界，内层判断跳过 Alpha 通道这两个判断解决的是不同问题，不能互相替代

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void invert_kernel(
    unsigned char* image,
    int width,
    int height
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 4;

    if (id < total && id % 4 != 3) {
        image[id] = 255 - image[id];
    }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(
    unsigned char* image,
    int width,
    int height
) {
    int threadsPerBlock = 256;
    int total = width * height * 4;
    int blocksPerGrid =
        (total + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```
