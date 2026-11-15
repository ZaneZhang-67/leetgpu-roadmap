---
id: attention-with-sinks
leetgpuId: 112
titleEn: Attention with Sinks
titleZh: 带注意力汇聚点的滑动窗口注意力
description: 使用 CUDA 实现同时保留因果汇聚点与局部窗口的稳定 Softmax 注意力
problemUrl: https://leetgpu.com/challenges/attention-with-sinks
updatedAt: 2026-11-12
---
## 题目定义

`Q/K/V/output` 均为 `[M,d]`。对查询位置 `i`，键位置 `j` 仅在以下条件下可见：

```text
j ≤ i 且 (j < num_sinks 或 j ≥ i-window_size+1)
```

在该可见集合上计算 `QKᵀ/√d`、Softmax 和对 V 的加权和。sink 仍受因果条件约束，未来 sink 不可见；sink 与窗口重叠时只计一次。

### 实现要求

- 保持 `solve` 签名不变，仅使用原生功能
- Softmax 只归一化可见位置
- 结果写入 `[M,d]` 的 `output`

### 示例

官方示例为 `M=4,d=4,num_sinks=1,window_size=2`。位置 3 可见 sink 0 和窗口位置 2、3。

### 约束

- `1≤M≤10,000`，`1≤d≤128`
- `1≤num_sinks≤min(16,M)`，`1≤window_size≤M`
- Q、K、V 和 output 均为 `[M,d]` FP32，输入值位于 `[-1,1]`
- 功能测试覆盖单 token、窗口覆盖完整前缀、`window_size=1`、非 2 的幂尺寸
- 性能测试为 `M=5000,d=128,num_sinks=4,window_size=1024`
- 绝对误差和相对误差均为 `1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(
    const float* Q, const float* K, const float* V,
    float* output, int M, int d,
    int num_sinks, int window_size
) {}
```

starter 未提供中间 score 或 probability 缓冲区。

## 实现思路

一个 Block 处理一个查询行，128 个线程协作计算每个可见键的 score；线程分别保存自己的输出维累加器。每读入一个键，线程 0 更新在线 Softmax 的最大值与分母，所有线程用同一对缩放因子更新分子，因此无需随窗口长度增长的权重缓冲。

可见键按升序生成：先加入不在窗口内的历史 sink，再加入窗口与因果前缀的交集，从而避免重复。归约槽、分母和分子都用 FP64，Block 的 Shared Memory 占用固定，不受最大窗口 10000 影响。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void sink_attention(
    const float* Q, const float* K, const float* V,
    float* output, int d, int num_sinks, int window_size
) {
    __shared__ double reduction[128];
    __shared__ double maximum, denominator, old_factor, new_weight;
    int row = static_cast<int>(blockIdx.x);
    int tid = static_cast<int>(threadIdx.x);
    int windowStart = max(0, row - window_size + 1);
    int sinkEnd = min(min(num_sinks, windowStart), row + 1);
    int count = sinkEnd + row - windowStart + 1;
    double scale = 1.0 / sqrt(static_cast<double>(d));
    double numerator = 0.0;
    if (tid == 0) { maximum = -CUDART_INF; denominator = 0.0; }
    __syncthreads();

    for (int p = 0; p < count; ++p) {
        int key = p < sinkEnd ? p : windowStart + p - sinkEnd;
        reduction[tid] = tid < d
            ? static_cast<double>(Q[row*d+tid]) * K[key*d+tid] : 0.0;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) reduction[tid] += reduction[tid + off];
            __syncthreads();
        }
        if (tid == 0) {
            double score = reduction[0] * scale;
            double nextMaximum = fmax(maximum, score);
            old_factor = exp(maximum - nextMaximum);
            new_weight = exp(score - nextMaximum);
            denominator = denominator * old_factor + new_weight;
            maximum = nextMaximum;
        }
        __syncthreads();
        if (tid < d)
            numerator = numerator * old_factor
                      + new_weight * static_cast<double>(V[key*d+tid]);
        __syncthreads();
    }
    if (tid < d) output[row*d+tid] = static_cast<float>(numerator / denominator);
}

extern "C" void solve(
    const float* Q, const float* K, const float* V,
    float* output, int M, int d,
    int num_sinks, int window_size
) {
    int threads = 128;
    sink_attention<<<M, threads>>>(Q,K,V,output,d,num_sinks,window_size);
    cudaDeviceSynchronize();
}
```
