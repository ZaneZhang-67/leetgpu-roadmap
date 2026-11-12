---
id: rotary-positional-embedding
leetgpuId: 61
titleEn: Rotary Positional Embedding
titleZh: 旋转位置编码
description: 使用 CUDA 对 M×D 查询矩阵执行 half-split RoPE，理解半维配对和位置表布局
problemUrl: https://leetgpu.com/challenges/rotary-positional-embedding
updatedAt: 2026-11-12
---
## 题目定义

给定形状均为 `(M, D)` 的 FP32 矩阵 `Q`、`cos` 和 `sin`，计算旋转位置编码：

```text
output = Q × cos + rotate_half(Q) × sin
```

`D` 保证为偶数若一行向量分为前后两半 `x1`、`x2`，则：

```text
rotate_half([x1, x2]) = [-x2, x1]
```

`cos` 与 `sin` 使用 half-split 布局：同一行中第 `j` 维和第 `j + D/2` 维的值相同

### 示例

```text
输入：Q = [[1.0, 2.0, 3.0, 4.0],
           [1.0, 1.0, 1.0, 1.0]]
      Cos = [[1.0, 1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0]]
      Sin = [[0.0, 0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0, 1.0]]
输出：[[1.0, 2.0, 3.0, 4.0],
       [-1.0, -1.0, 1.0, 1.0]]
```

第一行由 `Cos` 保持不变，第二行由 `Sin` 选择旋转后的向量。

**实现要求与测试**

- 不允许使用外部库，`solve` 签名保持不变
- 结果写入形状为 `(M,D)` 的 `output`
- `1 ≤ M,D ≤ 10,000`，`D` 为偶数
- 功能测试包含 `D=2`、全零、正负混合值和 `256×128`
- runtime 性能测试额外使用 `M=1,048,576`、`D=128`，其中 `M` 高于题面文字上限，实现须按真实性能规模覆盖
- 绝对与相对误差容限均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin,
                      float* output, int M, int D) {}
```

四个矩阵都按连续行主序存储，元素 `(row,col)` 的线性下标是 `row*D+col`。模板没有提供 kernel 和启动配置。

## 实现思路

一个线程处理一个输出元素由线性下标得到列号 `col=index%D`，再根据它处于哪一半定位配对元素：

```text
col < D/2：rotated = -Q[row, col+D/2]
否则：       rotated =  Q[row, col-D/2]
```

随后计算 `Q[index]*cos[index] + rotated*sin[index]`。`cos` 和 `sin` 的位置维已经展开为 `(M,D)`，不需要按 token 重新生成角度，也不能采用相邻偶奇维配对。

正式性能测试的总元素数超过一亿，因此使用 grid-stride 循环和 `size_t` 索引，避免窄化。每个输出元素独立计算，不需要同步或临时空间。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void rope_kernel(float* Q, float* cos,
                            float* sin, float* output,
                            size_t total, int D) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    int half = D / 2;

    for (; index < total; index += stride) {
        int col = (int)(index % D);
        size_t pair = col < half ? index + half : index - half;
        float rotated = col < half ? -Q[pair] : Q[pair];
        output[index] = Q[index] * cos[index] + rotated * sin[index];
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin,
                      float* output, int M, int D) {
    size_t total = (size_t)M * D;
    int threads = 256;
    size_t needed = (total + threads - 1) / threads;
    int blocks = needed > 4096 ? 4096 : (int)needed;
    rope_kernel<<<blocks, threads>>>(Q, cos, sin, output, total, D);
    cudaDeviceSynchronize();
}
```
