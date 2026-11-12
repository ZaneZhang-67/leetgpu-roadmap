---
id: sigmoid-linear-unit
leetgpuId: 52
titleEn: Sigmoid Linear Unit
titleZh: SiLU 激活函数
description: 使用 CUDA 对一维浮点向量逐元素计算 SiLU，理解 Sigmoid 与输入门控的组合
problemUrl: https://leetgpu.com/challenges/sigmoid-linear-unit
updatedAt: 2026-11-12
---
## 题目定义

对形状为 `[N]` 的一维浮点输入逐元素执行 SiLU（Sigmoid Linear Unit）激活函数：

```text
σ(x) = 1 / (1 + e^(-x))
SiLU(x) = x × σ(x)
```

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `output`

### 示例 1

```text
输入：input = [0.5, 1.0, -0.5]，N = 3
输出：output = [0.3112295, 0.731059, -0.1887705]
```

### 示例 2

```text
输入：input = [-1.0, -2.0, -3.0, -4.0, -5.0]，N = 5
输出：output = [-0.26894143, -0.23840584, -0.14227763, -0.07194484, -0.03346425]
```

### 约束

- `1 ≤ N ≤ 10,000`
- 输入值范围为 `[-100.0, 100.0]`
- 性能测试使用 `N = 50,000`

## 模板代码解析

平台 CUDA starter 如下：

```cpp
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

`input` 和 `output` 是 GPU 显存指针，`N` 表示元素数量模板按每个 Block 256 个线程配置一维 Grid，并通过向上取整覆盖全部元素需要补全 `silu_kernel` 函数体

## 实现思路

### 1. 一个线程对应一个元素

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

线程 `i` 读取 `input[i]` 并写入 `output[i]`

### 2. 检查边界

```cpp
if (i < N)
```

这个判断防止最后一个 Block 中的多余线程访问数组之外的地址

### 3. 计算 SiLU

CUDA 的 `expf` 计算单精度指数：

```cpp
float x = input[i];
float sigmoid = 1.0f / (1.0f + expf(-x));
output[i] = x * sigmoid;
```

输入与输出都是连续的一维 `float` 数组相邻线程访问相邻元素，各线程之间没有数据依赖

### 4. 覆盖完整输入范围

题目输入可达到 `[-100, 100]`使用上述直接公式时，极大负数会使 `expf(-x)` 溢出为正无穷，计算结果随之下溢为带符号零；数学上的 SiLU 仍是极小负数，但两者差异远低于题目容差正大值时 Sigmoid 趋近 1，SiLU 趋近原输入

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
