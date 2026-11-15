---
id: causal-depthwise-conv1d
leetgpuId: 90
titleEn: Causal Depthwise Conv1d
titleZh: 因果深度可分离一维卷积
description: 在 channels-last 序列上按通道独立执行只访问当前与历史位置的一维卷积
problemUrl: https://leetgpu.com/challenges/causal-depthwise-conv1d
updatedAt: 2026-11-12
---
## 题目定义

输入与输出的形状均为 `[B,L,D]`，采用 channels-last 布局。每个通道使用各自长度为 `K` 的卷积核，通道之间互不混合：

```text
output[b,l,d] = bias[d] + Σ weight[d,k] * x[b,l-k,d]
```

当 `l-k<0` 时，输入视为 0。`weight[d,0]` 乘当前位置，`weight[d,K-1]` 乘最旧位置。这与 runtime 先在左侧填充 `K-1` 个 0，再翻转 weight 并调用 cross-correlation 完全一致。

### 官方示例

```text
B=1,L=4,D=2,K=3
x=[[[1,2],[3,4],[5,6],[7,8]]]
weight=[[1,0,-1],[1,1,1]]
bias=[0,0]
output=[[[1,2],[3,6],[4,12],[4,18]]]
```

### 完整约束

- `1 ≤ B ≤ 16`，`1 ≤ L,D ≤ 8192`
- `1 ≤ K ≤ 8`
- 所有张量均为 FP32
- `x` 与 `output` 使用 `[B,L,D]` channels-last 布局
- 性能测试为 `B=8,L=2048,D=4096,K=4`
- 容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight,
                      const float* bias, float* output,
                      int B, int L, int D, int K) {}
```

`weight` 的形状为 `[D,K]`，`bias` 的形状为 `[D]`。参数 `D` 表示通道数，内存中连续存放的是通道维。

## 实现思路

每个线程负责一个 `(b,l,d)` 输出。线程从线性下标依次还原 `d=index%D`、`l=(index/D)%L` 和 `b=index/(L*D)`。

循环遍历 `k=0..K-1` 并读取 `x[b,l-k,d]`；索引越过左边界时，跳过该项。这样可以直接实现题面规定的卷积方向。若写成 `x[l-K+1+k]*weight[k]`，还必须对权重作相应翻转。

每个线程最多执行 8 次乘加。Grid-Stride Loop 在限制 Grid 数量的同时，可以覆盖性能规模中的 6710 万个输出。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void causal_depthwise(const float* x, const float* weight,
                                 const float* bias, float* output,
                                 long long total, int L, int D, int K) {
    long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; index < total; index += stride) {
        int d = static_cast<int>(index % D);
        long long position_index = index / D;
        int l = static_cast<int>(position_index % L);
        int b = static_cast<int>(position_index / L);
        float sum = bias[d];
        for (int k = 0; k < K; ++k) {
            int source = l - k;
            if (source >= 0) {
                long long xi = (static_cast<long long>(b) * L + source) * D + d;
                sum += weight[d * K + k] * x[xi];
            }
        }
        output[index] = sum;
    }
}

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight,
                      const float* bias, float* output,
                      int B, int L, int D, int K) {
    long long total = static_cast<long long>(B) * L * D;
    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    causal_depthwise<<<blocks, threads>>>(x, weight, bias, output,
                                          total, L, D, K);
    cudaDeviceSynchronize();
}
```
