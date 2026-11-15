---
id: matrix-copy
leetgpuId: 31
titleEn: Matrix Copy
titleZh: 矩阵复制
description: 使用 CUDA 将 N×N 浮点矩阵逐元素复制到输出矩阵，练习二维数据的一维线性映射
problemUrl: https://leetgpu.com/challenges/matrix-copy
updatedAt: 2026-11-12
---
## 题目定义

在 GPU 上将一个 `N × N` 的 32 位浮点矩阵从输入数组 `A` 复制到输出数组 `B`，使每个合法位置都满足：

```text
B[i,j] = A[i,j]
```

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入矩阵 `B`

### 示例 1

```text
输入：A = [[1.0, 2.0],
           [3.0, 4.0]]
输出：B = [[1.0, 2.0],
           [3.0, 4.0]]
```

### 示例 2

```text
输入：A = [[5.5, 6.6, 7.7],
           [8.8, 9.9, 10.1],
           [11.2, 12.3, 13.4]]
输出：B = [[5.5, 6.6, 7.7],
           [8.8, 9.9, 10.1],
           [11.2, 12.3, 13.4]]
```

### 约束

- `1 ≤ N ≤ 4096`
- 所有元素都是 32 位浮点数
- 性能测试使用 `N = 4096`

## 模板代码解析

平台提供空的复制 kernel 和 CPU 端启动代码：

```cpp
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
```

`A` 和 `B` 都指向 GPU 显存，`A` 带有 `const`，表示 kernel 只读取输入`solve` 根据矩阵边长计算元素总数 `N × N`，每个 Block 启动 256 个线程，并按总元素数向上取整需要补全的部分只有 `copy_matrix_kernel` 函数体

## 实现思路

### 1. 将矩阵视为连续数组

行主序矩阵在显存中是一段长度为 `total = N × N` 的连续 `float` 数组逐元素复制不需要显式计算行列坐标

### 2. 映射线程下标

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

线程 `i` 负责线性位置 `A[i]` 和 `B[i]`

### 3. 检查边界

Block 数向上取整后可能产生多余线程，因此访问数组前判断：

```cpp
if (i < total)
```

### 4. 复制元素

```cpp
B[i] = A[i];
```

相邻线程读写相邻元素，不同线程之间没有数据依赖，也不需要同步如果需要对应二维坐标，可以用 `row = i / N`、`col = i % N` 推导，但复制本身无需这些计算

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total) {
        B[i] = A[i];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
```
