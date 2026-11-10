---
id: matrix-transpose
titleEn: Matrix Transpose
titleZh: 矩阵转置
description: 使用原生 CUDA 转置行主序浮点矩阵，介绍二维线程映射、输入输出坐标交换、行主序索引和边界判断
problemUrl: https://leetgpu.com/challenges/matrix-transpose
updatedAt: 2026-11-10
---
## 题目定义

编写一个 GPU 程序，转置一个由 32 位浮点数组成的矩阵

输入矩阵 `A` 的形状为 `rows × cols`，转置后的矩阵 `Aᵀ` 形状为 `cols × rows`矩阵转置会交换元素的行、列坐标：

```text
output[col, row] = input[row, col]
```

输入和输出矩阵都采用行主序存储

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入矩阵 `output`

### 示例 1

```text
输入：2 × 3 矩阵

[1.0  2.0  3.0]
[4.0  5.0  6.0]

输出：3 × 2 矩阵

[1.0  4.0]
[2.0  5.0]
[3.0  6.0]
```

### 示例 2

```text
输入：3 × 1 矩阵

[1.0]
[2.0]
[3.0]

输出：1 × 3 矩阵

[1.0  2.0  3.0]
```

### 约束

- `1 ≤ rows, cols ≤ 8192`
- 输入矩阵形状为 `rows × cols`
- 输出矩阵形状为 `cols × rows`
- 性能测试使用 `cols = 6000`、`rows = 7000`

## 模板代码解析

模板已经提供 kernel 声明和 CPU 端入口函数 `solve`，需要补全 `matrix_transpose_kernel` 的函数体

```cpp
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, rows, cols
    );
    cudaDeviceSynchronize();
}
```

参数含义：

- `input`：输入矩阵，形状为 `rows × cols`
- `output`：输出矩阵，形状为 `cols × rows`
- `rows`：输入矩阵的行数，也是输出矩阵的列数
- `cols`：输入矩阵的列数，也是输出矩阵的行数

`input` 和 `output` 都指向 GPU 显存，`input` 带有 `const`，kernel 只读取它

`threadsPerBlock(16, 16)` 表示每个 Block 含有 `16 × 16 = 256` 个线程Block 的 `x` 方向覆盖输入矩阵的列，`y` 方向覆盖输入矩阵的行

Grid 在两个方向上分别向上取整：

```text
Grid.x = ceil(cols / 16)
Grid.y = ceil(rows / 16)
```

这样启动的线程可以覆盖输入矩阵的全部 `rows × cols` 个元素

## 实现思路

朴素转置让一个线程负责输入矩阵中的一个元素线程读取 `input[row, col]`，再把它写入输出矩阵的 `output[col, row]`

### 1. 计算输入坐标

二维 Grid 中，线程对应的输入坐标为：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

其中，`row` 对应输入矩阵的行，合法范围是 `[0, rows)`；`col` 对应输入矩阵的列，合法范围是 `[0, cols)`

### 2. 检查边界

Grid 的两个维度都使用向上取整，边缘 Block 可能包含没有对应输入元素的线程访问矩阵前需要判断：

```cpp
if (row < rows && col < cols)
```

缺少这个判断时，多余线程可能访问输入或输出矩阵之外的显存

### 3. 推导输入地址

输入矩阵形状为 `rows × cols`，每行有 `cols` 个元素行主序下，`input[row, col]` 的线性下标为：

```cpp
int inputIndex = row * cols + col;
```

例如，一个 `2 × 3` 矩阵在显存中的排列为：

```text
input = [1, 2, 3, 4, 5, 6]

坐标：   (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
下标：      0     1     2     3     4     5
```

### 4. 交换行列并推导输出地址

转置后，输入坐标 `(row, col)` 变成输出坐标 `(col, row)`输出矩阵形状为 `cols × rows`，因此输出矩阵每行有 `rows` 个元素，而不是 `cols` 个元素

行主序下，`output[col, row]` 的线性下标为：

```cpp
int outputIndex = col * rows + row;
```

最终赋值为：

```cpp
output[col * rows + row] = input[row * cols + col];
```

以示例 1 中的 `input[1,2] = 6` 为例：

```text
输入下标 = 1 × 3 + 2 = 5
输出坐标 = (2,1)
输出下标 = 2 × 2 + 1 = 5
```

矩阵中的数值没有变化，只有它在二维视图中的行列位置发生了交换

### 5. 线程映射

```text
线程 (row=0, col=0) → output[0,0] = input[0,0]
线程 (row=0, col=1) → output[1,0] = input[0,1]
线程 (row=0, col=2) → output[2,0] = input[0,2]
线程 (row=1, col=0) → output[0,1] = input[1,0]
线程 (row=1, col=1) → output[1,1] = input[1,1]
线程 (row=1, col=2) → output[2,1] = input[1,2]
```

每个线程只处理一个元素，不同线程写入不同的输出下标，因此不需要在 kernel 内同步

这个实现优先说明转置的坐标和地址关系它读取 `input` 时，相邻线程访问相邻元素；写入 `output` 时，相邻线程落在不同的输出行，访问地址不连续后续可以使用 Shared Memory tile 改善输出写入模式，但不影响本题朴素实现的索引公式

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, rows, cols
    );
    cudaDeviceSynchronize();
}
```
