---
id: min-p-sampling
leetgpuId: 104
titleEn: Min-P Sampling
titleZh: Min-P 采样概率过滤
description: 按每行最大概率的相对阈值过滤 logits，并重新归一化保留项
problemUrl: https://leetgpu.com/challenges/min-p-sampling
updatedAt: 2026-11-12
---
## 题目定义

输入为 `logits[B,V]`，输出同形状的过滤概率。每一行先减去最大 logit，再计算 `e_i=exp(logit_i-max_logit)`。Softmax 分母会在相对阈值两边抵消，因此 runtime 直接保留 `e_i>=min_p` 的项，最后对保留项重新归一化：

```text
keep_i = exp(logit_i-max_logit) >= min_p
probs_i = keep_i ? exp(logit_i-max_logit)/Σkept : 0
```

最大项的指数恒为 1，因此 `min_p≤1` 时至少会保留一项。题目只输出概率，不接收 temperature、随机数或 CDF，也不返回采样 token；并列的最大项都会保留。

### 官方示例

```text
B=2,V=4,min_p=0.1
logits=[[1,2,3,4],[-1,0,1,-2]]
probs=[[0,0.0900,0.2447,0.6652],
       [0.0900,0.2447,0.6652,0]]
```

### 完整约束

- `1 ≤ B ≤ 256`，`1 ≤ V ≤ 200000`
- `0 ≤ min_p ≤ 1`
- `-50 ≤ logits[b,i] ≤ 50`
- 输入输出均为 FP32
- 每行独立过滤，屏蔽项必须精确为 0，输出行重新归一化
- 性能测试为 `B=64,V=128000,min_p=0.05`
- 判定容差为 `1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// logits, probs are device pointers
extern "C" void solve(const float* logits, float* probs,
                      float min_p, int B, int V) {}
```

`min_p` 表示相对于行最大概率的比例，既不是累计概率阈值，也不是绝对概率 cutoff，因此无需排序。

## 实现思路

一个 Block 负责一行，256 个线程共同扫描词表。第一次并行归约得到最大 logit；第二次计算指数，按 `e>=min_p` 过滤并归约保留项总和；第三次再并行写回归一化概率。

第一次归约使指数输入不大于 0。第二次把指数暂存到 `probs`，第三次便无需再次调用 `expf`。每行由 256 个线程并行覆盖最多 20 万词表，避免使用单线程大词表循环。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void min_p_kernel(const float* logits, float* probs,
                             float min_p, int V) {
    __shared__ float shared[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < V; i += blockDim.x)
        local_max = fmaxf(local_max, logits[row * V + i]);
    shared[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        __syncthreads();
    }
    float maximum = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) {
        float value = expf(logits[row * V + i] - maximum);
        if (value < min_p) value = 0.0f;
        probs[row * V + i] = value;
        local_sum += value;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    float denominator = shared[0];
    for (int i = tid; i < V; i += blockDim.x)
        probs[row * V + i] /= denominator;
}

extern "C" void solve(const float* logits, float* probs,
                      float min_p, int B, int V) {
    min_p_kernel<<<B, 256>>>(logits, probs, min_p, V);
    cudaDeviceSynchronize();
}
```
