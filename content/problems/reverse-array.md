---
id: reverse-array
leetgpuId: 19
titleEn: Reverse Array
titleZh: 数组反转
description: 使用 CUDA 原地反转 32 位浮点数组，理解成对交换、对称下标与半区线程映射
problemUrl: https://leetgpu.com/challenges/reverse-array
updatedAt: 2026-11-12
---
## 题目定义

原地反转一个由 32 位浮点数组成的数组 `input`反转后，第一个元素与最后一个元素交换，第二个元素与倒数第二个元素交换，依次进行

### 实现要求

- 只使用原生 CUDA 功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写回 `input`

### 示例 1

```text
输入：[1.0, 2.0, 3.0, 4.0]
输出：[4.0, 3.0, 2.0, 1.0]
```

### 示例 2

```text
输入：[1.5, 2.5, 3.5]
输出：[3.5, 2.5, 1.5]
```

### 约束

- `1 ≤ N ≤ 100,000,000`
- 性能测试使用 `N = 25,000,000`

## 模板代码解析

平台提供空的 `reverse_array` kernel 和 CPU 端入口：

```cpp
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
```

`input` 是指向 GPU 显存的可写指针，`N` 是数组长度模板以 256 个线程组成一个 Block，并按 `N` 向上取整计算 Block 数，因此启动的线程足以覆盖整个数组需要补全的部分只有 kernel 函数体

## 实现思路

原地反转不能让每个线程都执行一次交换，否则一对元素会被交换两次，最终回到原位只让前半段线程工作即可

### 1. 计算线程下标

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

线程 `i` 负责数组左侧下标 `i`

### 2. 推导对称位置

长度为 `N` 的数组，`i` 对应的右侧下标为：

```cpp
int j = N - 1 - i;
```

例如 `N = 6` 时，交换对为 `(0,5)`、`(1,4)`、`(2,3)`

### 3. 限制工作线程

只处理：

```cpp
if (i < N / 2)
```

偶数长度会交换全部元素；奇数长度的中间元素无需移动，因此整除结果仍然正确模板虽然按 `N` 启动线程，但后半段线程会直接退出

### 4. 原地交换

```cpp
float temp = input[i];
input[i] = input[j];
input[j] = temp;
```

每个有效线程处理一对互不重叠的元素，不需要线程同步

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N / 2) {
        int j = N - 1 - i;
        float temp = input[i];
        input[i] = input[j];
        input[j] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
```
