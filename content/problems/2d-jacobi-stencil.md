---
id: 2d-jacobi-stencil
leetgpuId: 69
titleEn: 2D Jacobi Stencil
titleZh: 二维 Jacobi 模板计算
description: 使用 CUDA 对二维矩阵执行一次五点 Jacobi 更新，并保持边界元素不变
problemUrl: https://leetgpu.com/challenges/2d-jacobi-stencil
updatedAt: 2026-11-12
---
## 题目定义

给定 FP32 行主序矩阵 `input(rows,cols)`，执行一次五点 Jacobi 更新。内部元素取上下左右四个邻居的平均值：

```text
output[r,c] = 0.25 × (input[r-1,c] + input[r+1,c]
                    + input[r,c-1] + input[r,c+1])
```

第一行、最后一行、第一列和最后一列属于边界，必须原样复制。题目只要求一次更新，不进行多轮迭代，也不需要交换输入输出缓冲。

### 示例

```text
输入（4×4）：
[ 1  2  3  4]
[ 5  6  7  8]
[ 9 10 11 12]
[13 14 15 16]

输出（4×4）：
[ 1  2  3  4]
[ 5  6  7  8]
[ 9 10 11 12]
[13 14 15 16]
```

例如，内部元素 `(1,1)` 为 `0.25×(2+10+5+7)=6`。其余内部元素也恰好等于对应的原值，边界则直接复制。

约束为 `1≤rows,cols≤16384`，输入与输出均为 FP32，输入值位于 `[-100,100]`。官方测试覆盖 `1×1`、单行、单列、非方阵和非二次幂尺寸；性能用例为 `8192×8192`，容差为 `1e-5`。

## 模板代码解析

CUDA starter 的签名为：

```text
extern "C" void solve(const float* input, float* output, int rows, int cols)
```

两个数组都是设备指针，按 `row*cols+col` 寻址。`challenge.py` 先把输入复制到输出，再只覆盖内部区域，因此边界语义是复制而不是补零参与平均。

## 实现思路

使用二维 Grid，一个线程负责一个元素。线程先检查 `row<rows && col<cols`；若位于边界，直接复制 `input[index]`，否则读取四个合法邻居。这样 `rows<3` 或 `cols<3` 时所有元素自然进入边界分支。

每个输出只依赖只读输入，因此无需同步或临时缓冲，单次 kernel 即可覆盖最大矩阵。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void jacobi_2d_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;

    int index = row * cols + col;
    if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
        output[index] = input[index];
    } else {
        output[index] = 0.25f * (input[index - cols] + input[index + cols]
                               + input[index - 1] + input[index + 1]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);
    jacobi_2d_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```
