---
id: matrix-power
leetgpuId: 37
titleEn: Matrix Power
titleZh: 矩阵幂
description: 使用原生 CUDA 计算方阵的正整数次幂，理解矩阵乘法、幂语义、设备端临时缓冲与多轮 kernel 调度
problemUrl: https://leetgpu.com/challenges/matrix-power
updatedAt: 2026-11-12
---
## 题目定义

输入是按行主序展平的 `N × N` 方阵 `input`，要求计算标准稠密矩阵乘法意义下的 `A^P`，并把结果写入同样按行主序存储的 `output`

`P` 是正整数，因此：

```text
P = 1：output = A
P = 2：output = A × A
P = 3：output = A × A × A
```

这里的幂不是逐元素乘方每次乘法都需要按行列内积计算

### 实现要求与约束

- 不允许使用外部库
- `solve` 函数签名必须保持不变
- 结果写入 `output`
- `1 ≤ N ≤ 1024`
- `1 ≤ P ≤ 20`
- `-10.0 ≤ input[i] ≤ 10.0`
- 性能测试使用 `N = 512`、`P = 3`

### 示例

```text
输入：A = [[1, 2],
           [3, 4]]
      P = 3
输出：[[37, 54],
       [81, 118]]
```

计算顺序是先求 `A²`，再求 `A² × A`

## 模板代码解析

CUDA starter 只提供固定的 CPU 入口：

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {}
```

`input` 和 `output` 已经是设备指针平台没有提供额外工作区，而连续矩阵乘法不能一直覆盖同一输入，因此 `solve` 需要自行申请至少一个 `N × N` 临时矩阵

参考实现采用朴素的重复乘法：先把 `input` 复制到 `output`，随后执行 `P - 1` 次 `output × input`每轮把乘积写入临时缓冲，再交换“当前结果”和“目标缓冲”的指针最后若当前结果不在 `output`，再复制一次

`P = 1` 必须单独成立：复制输入后不再启动矩阵乘法 kernel

## 实现思路

### 1. 一个线程计算一个输出元素

线程坐标 `(row, col)` 对应乘积矩阵的一个元素：

```cpp
sum = 0;
for (int k = 0; k < N; ++k) {
    sum += left[row * N + k] * right[k * N + col];
}
out[row * N + col] = sum;
```

二维 Grid 在两个方向向上取整，kernel 内检查 `row < N && col < N`

### 2. 临时缓冲与指针交换

覆盖 `output` 会破坏本轮仍需读取的旧结果，所以每轮必须读写分离：

```text
当前结果 current ──× input──> next
交换 current 与 next
```

只申请一个临时矩阵即可，因为 `input` 始终只读，而当前结果在 `output` 与临时缓冲之间交替

### 3. 复杂度

朴素矩阵乘法每轮执行 `N³` 次乘加，总计 `P - 1` 轮，时间复杂度为 `O(PN³)`，额外空间为 `O(N²)`这份实现首先保证 starter 签名和全部功能测试正确；更高性能的版本可使用 tiled GEMM 或二进制快速幂，但快速幂仍需要妥善管理多个矩阵缓冲

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void matmul(const float* left, const float* right, float* out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += left[row * N + k] * right[k * N + col];
    }
    out[row * N + col] = sum;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
    if (P == 1) {
        cudaDeviceSynchronize();
        return;
    }

    float* temp = nullptr;
    cudaMalloc(&temp, bytes);
    float* current = output;
    float* next = temp;

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    for (int power = 2; power <= P; ++power) {
        matmul<<<blocks, threads>>>(current, input, next, N);
        float* swap = current;
        current = next;
        next = swap;
    }

    if (current != output) {
        cudaMemcpy(output, current, bytes, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    cudaFree(temp);
}
```
