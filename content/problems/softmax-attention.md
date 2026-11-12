---
id: softmax-attention
leetgpuId: 6
titleEn: Softmax Attention
titleZh: Softmax 注意力
description: 使用 CUDA 直接计算缩放点积注意力，为 Q 的每一行生成稳定的 Softmax 权重，并对 V 做加权求和
problemUrl: https://leetgpu.com/challenges/softmax-attention
updatedAt: 2026-11-12
---
## 题目定义

给定查询矩阵 `Q`、键矩阵 `K` 和值矩阵 `V`：

```text
Q：M × d
K：N × d
V：N × d
```

计算缩放点积注意力：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

Softmax 按行计算，最终输出矩阵形状为 `M × d`

对于查询行 `row` 和键行 `keyRow`，注意力分数为：

```text
score[row, keyRow]
= dot(Q[row, :], K[keyRow, :]) / √d
```

对同一查询行的 `N` 个分数执行 Softmax 后，再使用所得权重对 `V` 的各行加权求和

### 实现要求

- 只使用 GPU 原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入矩阵 `output`

### 示例 1

```text
Q（2 × 4）：
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0]]

K（3 × 4）：
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0]]

V（3 × 4）：
[[1.0, 2.0, 3.0, 4.0],
 [5.0, 6.0, 7.0, 8.0],
 [9.0, 10.0, 11.0, 12.0]]

输出（2 × 4）：
[[4.29, 5.29, 6.29, 7.29],
 [5.00, 6.00, 7.00, 8.00]]
```

### 示例 2

```text
Q（1 × 2）：[[1.0, 2.0]]
K（2 × 2）：[[1.0, 0.0],
             [0.0, 1.0]]
V（2 × 2）：[[3.0, 4.0],
             [5.0, 6.0]]

输出（1 × 2）：[[4.34, 5.34]]
```

### 约束

- `Q` 的形状为 `M × d`
- `K` 和 `V` 的形状均为 `N × d`
- `1 ≤ M, N ≤ 100,000`
- `1 ≤ d ≤ 128`
- 性能测试使用 `M = 512`、`N = 256`

## 模板代码解析

平台提供的 CUDA 模板只有固定的 `solve` 入口：

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M,
    int N,
    int d
) {}
```

- `Q` 指向行主序的 `M × d` 查询矩阵
- `K` 指向行主序的 `N × d` 键矩阵
- `V` 指向行主序的 `N × d` 值矩阵
- `output` 指向行主序的 `M × d` 输出矩阵
- `M` 是查询行数，`N` 是键和值的行数，`d` 是每行的特征维度

模板没有提供 kernel、临时存储或启动配置需要在 `solve` 内完成 kernel 启动，并保持参数顺序和类型不变

## 实现思路

参考实现不显式保存完整的 `M × N` 分数矩阵，而是让一个线程负责一个输出元素 `output[row, col]`该线程为查询行 `row` 扫描全部 `N` 个键，使用在线 Softmax 同时维护归一化分母和对 `V[:, col]` 的加权和

### 1. 输出线程映射

输出矩阵共有 `M × d` 个元素，线程通过一维下标映射到二维坐标：

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
int row = index / d;
int col = index % d;
```

只有 `index < M * d` 的线程才执行计算

### 2. 计算缩放点积分数

对于当前查询行 `row` 和键行 `keyRow`，点积为：

```cpp
float score = 0.0f;
for (int k = 0; k < d; ++k) {
    score += Q[row * d + k] * K[keyRow * d + k];
}
score /= sqrtf(static_cast<float>(d));
```

`Q[row * d + k]` 和 `K[keyRow * d + k]` 都是行主序地址公式

### 3. 在线稳定 Softmax

若先保存所有分数，需要额外分配 `M × N` 临时空间本实现按键行顺序扫描分数，并维护三个状态：

```text
runningMax：当前已处理分数的最大值
runningSum：相对于 runningMax 的指数和
weighted：当前输出列的加权和
```

读到新分数 `score` 后，新最大值为：

```cpp
float newMax = fmaxf(runningMax, score);
```

已有累计量需要重新缩放到 `newMax` 基准：

```cpp
float oldScale = expf(runningMax - newMax);
float newScale = expf(score - newMax);

runningSum = runningSum * oldScale + newScale;
weighted = weighted * oldScale + newScale * V[keyRow * d + col];
runningMax = newMax;
```

因为指数的自变量都不大于 0，这一更新同样使用了最大值技巧，可避免直接计算过大的指数

### 4. 得到归一化输出

扫描全部 `N` 个键后：

```cpp
output[row * d + col] = weighted / runningSum;
```

它等价于：

```text
Σ softmax(score)[keyRow] × V[keyRow, col]
```

同一查询行的不同输出列使用相同的注意力分数参考实现让各线程分别重算分数，以换取不分配全局临时缓冲区和不改变 `solve` 接口

### 5. 正确性与实现取舍

该实现直接得到最终注意力结果，不保存 `QKᵀ` 或 Softmax 矩阵，临时状态只存在于寄存器中

代价是同一查询行的 `d` 个输出线程会重复计算相同的 `Q·K` 点积性能更高的版本可以让一个 Block 协作处理一行查询，在共享内存中复用分数和权重；参考实现优先保持索引、数值稳定性和接口完整性清晰

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M,
    int N,
    int d
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * d;

    if (index >= total) return;

    int row = index / d;
    int col = index % d;
    float scale = rsqrtf(static_cast<float>(d));

    float runningMax = -CUDART_INF_F;
    float runningSum = 0.0f;
    float weighted = 0.0f;

    for (int keyRow = 0; keyRow < N; ++keyRow) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[row * d + k] * K[keyRow * d + k];
        }
        score *= scale;

        float newMax = fmaxf(runningMax, score);
        float oldScale = expf(runningMax - newMax);
        float newScale = expf(score - newMax);

        runningSum = runningSum * oldScale + newScale;
        weighted = weighted * oldScale + newScale * V[keyRow * d + col];
        runningMax = newMax;
    }

    output[row * d + col] = weighted / runningSum;
}

// Q, K, V, output are device pointers
extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M,
    int N,
    int d
) {
    constexpr int threadsPerBlock = 256;
    int total = M * d;
    int blocksPerGrid =
        (total + threadsPerBlock - 1) / threadsPerBlock;

    softmax_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        Q, K, V, output, M, N, d
    );
    cudaDeviceSynchronize();
}
```
