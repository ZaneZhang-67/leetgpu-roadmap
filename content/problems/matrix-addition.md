---
id: matrix-addition
titleEn: Matrix Addition
titleZh: 矩阵加法
description: 使用原生 CUDA 对两个 N×N 浮点矩阵执行逐元素加法，介绍矩阵线性化、一维线程映射、Grid 覆盖和边界判断
problemUrl: https://leetgpu.com/challenges/matrix-addition
updatedAt: 2026-11-12
---
## 题目定义

编写一个 GPU 程序，对两个包含 32 位浮点数的 `N × N` 矩阵执行逐元素加法输入矩阵 `A` 和 `B` 的形状相同，输出矩阵 `C` 保存对应位置元素之和：

```text
C[row, col] = A[row, col] + B[row, col]
```

所有矩阵均采用行主序存储

### 实现要求

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入矩阵 `C`

### 示例 1

```text
输入：A = [[1.0, 2.0],
           [3.0, 4.0]]
      B = [[5.0, 6.0],
           [7.0, 8.0]]
输出：C = [[6.0, 8.0],
           [10.0, 12.0]]
```

对应位置分别相加：

```text
C[0,0] = 1.0 + 5.0 = 6.0
C[0,1] = 2.0 + 6.0 = 8.0
C[1,0] = 3.0 + 7.0 = 10.0
C[1,1] = 4.0 + 8.0 = 12.0
```

### 示例 2

```text
输入：A = [[1.5, 2.5, 3.5],
           [4.5, 5.5, 6.5],
           [7.5, 8.5, 9.5]]
      B = [[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]]
输出：C = [[2.0, 3.0, 4.0],
           [5.0, 6.0, 7.0],
           [8.0, 9.0, 10.0]]
```

### 约束

- 输入矩阵 `A` 和 `B` 的形状相同
- `1 ≤ N ≤ 4096`
- 所有元素都是 32 位浮点数
- 性能测试使用 `N = 4096`

## 模板代码解析

模板提供了空的 GPU kernel 和完整的 CPU 端启动代码，需要补全 `matrix_add` 的函数体：

```cpp
#include <cuda_runtime.h>

__global__ void matrix_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

参数含义：

- `A`、`B`：输入矩阵，形状均为 `N × N`
- `C`：输出矩阵，形状为 `N × N`
- `N`：矩阵的行数和列数

`A`、`B`、`C` 都指向 GPU 显存，其中 `A` 和 `B` 带有 `const`，kernel 只读取它们

`threadsPerBlock = 256` 表示每个 Block 启动 256 个线程矩阵一共有 `N × N` 个元素，因此模板按元素总数计算 Block 数：

```cpp
int blocksPerGrid =
    (N * N + threadsPerBlock - 1) / threadsPerBlock;
```

这里使用一维 Grid，不需要显式创建二维线程坐标

## 实现思路

矩阵加法中，每个输出元素只依赖 `A`、`B` 中相同位置的两个元素，不同位置之间没有数据依赖因此，可以让一个 GPU 线程负责一个矩阵元素

### 1. 将矩阵视为一维连续数组

矩阵采用行主序存储，一个 `N × N` 矩阵在显存中包含 `N × N` 个连续元素

例如：

```text
矩阵：              显存中的一维布局：
[1  2  3]          [1, 2, 3, 4, 5, 6, 7, 8, 9]
[4  5  6]
[7  8  9]
```

逐元素加法不需要交换行列，也不需要访问相邻位置一个线程只要得到唯一的线性下标 `id`，就能读取并写入三个矩阵中的同一位置

### 2. 计算全局线程下标

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x`：当前 Block 在 Grid 中的编号
- `blockDim.x`：每个 Block 的线程数
- `threadIdx.x`：线程在当前 Block 内的编号

当每个 Block 有 256 个线程时，Block 0 负责前 256 个元素，Block 1 负责接下来的 256 个元素，依次类推

### 3. 为什么边界是 `N × N`

矩阵有 `N` 行，每行有 `N` 个元素，因此元素总数是：

```text
N × N
```

Grid 使用向上取整，最后一个 Block 可能包含没有对应矩阵元素的线程访问数组前必须检查：

```cpp
if (id < N * N)
```

缺少这个判断时，多余线程可能访问矩阵范围之外的显存

### 4. 执行逐元素加法

通过边界检查后，线程读取同一个线性位置上的两个输入值：

```cpp
C[id] = A[id] + B[id];
```

线程映射如下：

```text
线程 0  → C[0] = A[0] + B[0]
线程 1  → C[1] = A[1] + B[1]
线程 2  → C[2] = A[2] + B[2]
...
线程 id → C[id] = A[id] + B[id]
```

相邻线程访问相邻的 `float` 元素，读写地址连续不同线程写入 `C` 中不同的位置，没有写冲突，也不需要在 kernel 内同步

### 5. 线性下标与二维坐标的关系

虽然实现只使用 `id`，它仍然对应矩阵中的一个二维位置：

```cpp
int row = id / N;
int col = id % N;
```

反过来，二维坐标 `(row, col)` 的行主序下标为：

```cpp
int id = row * N + col;
```

本题只做相同位置的逐元素运算，所以没有必要在 kernel 中额外计算 `row` 和 `col`直接使用线性下标更简洁

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void matrix_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N * N) {
        C[id] = A[id] + B[id];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
