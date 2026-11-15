---
id: integral-image
leetgpuId: 77
titleEn: Integral Image
titleZh: 积分图
description: 使用 CUDA 计算二维 FP32 图像的横向与纵向前缀和
problemUrl: https://leetgpu.com/challenges/integral-image
updatedAt: 2026-11-12
---
## 题目定义

给定按行主序存储的 `H×W` FP32 图像，计算积分图：

```text
output[i,j] = Σ input[r,c]，其中 0≤r≤i，0≤c≤j
```

只能读取 `input`，不能原地修改它；结果写入 `output`。

### 示例

```text
input  = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
output = [[1, 3, 6],
          [5,12,21],
          [12,27,45]]
```

例如，`output[1,1]=1+2+4+5=12`。

**约束与测试**

- `1≤H,W≤16,384`，输入值位于 `[-100,100]`，输入输出均为 FP32
- 功能测试覆盖单元素、单行、单列、零矩阵、非二次幂和非方阵 `255×33`
- 性能测试为 `8192×8192`，绝对与相对误差容限均为 `1e-2`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W) {}
```

CUDA starter 中的 `solve` 为空，四个参数依次表示只读输入、输出以及行列数。作答时需要自行编写 kernel、分配必要的中间缓冲区并配置启动参数。

## 实现思路

二维积分图可以拆成两次一维扫描。为匹配 runtime reference 的浮点运算顺序，第一阶段沿 `dim=0` 对每一列求前缀和，第二阶段再沿 `dim=1` 对中间矩阵的每一行求前缀和。每行或每列分别由一个线程顺序扫描：列阶段有 `W` 个任务，行阶段有 `H` 个任务。单条扫描虽然串行，但各行、各列彼此独立，总工作量仍为 `O(HW)`。与为每个输出重新遍历左上矩形相比，这种方法可以覆盖 `8192×8192` 的正式规模。

第一阶段把列扫描结果写入临时缓冲区，第二阶段读取该缓冲区并写入 `output`，因此原始 `input` 始终只读。两阶段依次对应 `torch.cumsum(input,dim=0)` 和沿 `dim=1` 的 `cumsum`，全程使用 FP32。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void row_scan(const float* input, float* temp, int H, int W) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < H; row += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        size_t base = (size_t)row * W;
        for (int col = 0; col < W; ++col) {
            sum += input[base + col];
            temp[base + col] = sum;
        }
    }
}

__global__ void col_scan(const float* temp, float* output, int H, int W) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < W; col += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (int row = 0; row < H; ++row) {
            size_t index = (size_t)row * W + col;
            sum += temp[index];
            output[index] = sum;
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W) {
    float* temp;
    size_t bytes = (size_t)H * W * sizeof(float);
    cudaMalloc(&temp, bytes);
    int threads = 256;
    col_scan<<<(W + threads - 1) / threads, threads>>>(input, temp, H, W);
    row_scan<<<(H + threads - 1) / threads, threads>>>(temp, output, H, W);
    cudaDeviceSynchronize();
    cudaFree(temp);
}
```
