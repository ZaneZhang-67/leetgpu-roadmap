---
id: sliding-window-self-attention
leetgpuId: 59
titleEn: Sliding Window Self-Attention
titleZh: 滑动窗口自注意力
description: 将缩放点积注意力限制在查询位置左右 window_size 范围内，并以稳定 Softmax 聚合局部 Value
problemUrl: https://leetgpu.com/challenges/sliding-window-self-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q`、`K`、`V` 和 `output` 的形状均为 `[M,d]`对查询位置 `i`，只允许关注满足 `|j-i| ≤ window_size` 的 Key 与 Value：

```text
score(i,j) = dot(Q[i], K[j]) / sqrt(d)
weight(i,j) = softmax(score(i,*))
output[i] = Σ weight(i,j) × V[j]
```

窗口会在序列两端截断，合法范围是：

```text
begin = max(0, i-window_size)
end   = min(M-1, i+window_size)
```

这是双向局部注意力，不是因果掩码，查询可以同时关注窗口内的过去位置、当前位置和未来位置

### 实现要求与评测规模

- 不允许使用外部库
- `solve` 签名保持不变
- 最终结果写入 `output`
- `1 ≤ M ≤ 10,000`，`1 ≤ d ≤ 128`，`1 ≤ window_size ≤ 32`
- Q/K/V 元素范围为 `[-100,100]`
- 官方功能测试覆盖 `M=2,d=3/4`、零矩阵、混合正负值和 `M=128,d=32,window_size=8`
- 性能测试使用 `M=5000`、`d=64`、`window_size=16`
- 官方误差要求为 `atol=1e-5`、`rtol=1e-5`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K,
                      const float* V, float* output,
                      int M, int d, int window_size) {}
```

四个矩阵都已位于 GPU，平台只提供指针和三个整数参数作答需要实现局部打分、Softmax 和 Value 加权，不需要生成或存储完整的 `M × M` 注意力矩阵

## 实现思路

一个 Block 负责一个查询行 `i`为保证实现适用于任意 `d`，参考代码先由线程 0 按特征顺序计算局部 score、最大值与指数和，再把归一化权重保存在动态 Shared Memory 中这种计算顺序直接对应公式，也避免不同归约顺序在严格误差阈值下放大舍入差异

Softmax 使用减最大值形式：

```text
exp(score - maxScore)
```

可避免大正 score 使指数溢出每个窗口至少包含查询自身，因此指数和大于零

权重准备完毕后，Block 内线程并行处理输出特征 `feature=threadIdx.x, threadIdx.x+blockDim.x, ...`每个线程按窗口位置递增的固定顺序累计 `weight[j] * V[j,feature]`

动态 Shared Memory 只需容纳最多 `2*window_size+1` 个 `float`，性能用例中为 33 个元素计算量从全注意力的 `O(M²d)` 降为 `O(M·(2w+1)·d)`

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void sliding_attention(const float* Q, const float* K,
                                  const float* V, float* output,
                                  int M, int d, int window_size) {
    extern __shared__ float weights[];
    int row = blockIdx.x;
    int begin = max(0, row - window_size);
    int end = min(M - 1, row + window_size);
    int count = end - begin + 1;

    if (threadIdx.x == 0) {
        float scale = rsqrtf(static_cast<float>(d));
        float maximum = -CUDART_INF_F;
        for (int offset = 0; offset < count; ++offset) {
            int keyRow = begin + offset;
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[row * d + k] * K[keyRow * d + k];
            }
            score *= scale;
            weights[offset] = score;
            maximum = fmaxf(maximum, score);
        }

        float denominator = 0.0f;
        for (int offset = 0; offset < count; ++offset) {
            float value = expf(weights[offset] - maximum);
            weights[offset] = value;
            denominator += value;
        }
        for (int offset = 0; offset < count; ++offset) {
            weights[offset] /= denominator;
        }
    }
    __syncthreads();

    for (int feature = threadIdx.x; feature < d; feature += blockDim.x) {
        float sum = 0.0f;
        for (int offset = 0; offset < count; ++offset) {
            int valueRow = begin + offset;
            sum += weights[offset] * V[valueRow * d + feature];
        }
        output[row * d + feature] = sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K,
                      const float* V, float* output,
                      int M, int d, int window_size) {
    int threads = 256;
    size_t sharedBytes = static_cast<size_t>(2 * window_size + 1) * sizeof(float);
    sliding_attention<<<M, threads, sharedBytes>>>(Q, K, V, output,
                                                   M, d, window_size);
    cudaDeviceSynchronize();
}
```
