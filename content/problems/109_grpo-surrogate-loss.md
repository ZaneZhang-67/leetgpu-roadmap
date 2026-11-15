---
id: grpo-surrogate-loss
leetgpuId: 109
titleEn: GRPO Surrogate Loss
titleZh: GRPO 代理损失
description: 使用 CUDA 计算组内奖励标准化、逐 token 截断策略目标、参考策略 KL 惩罚及最终标量归约
problemUrl: https://leetgpu.com/challenges/grpo-surrogate-loss
updatedAt: 2026-11-12
---
## 题目定义

`rewards` 的形状为 `[B,G]`，三套 log probability 的形状为 `[B,G,S]`。先在每个 prompt 的 G 个响应之间计算总体标准差：

```text
mean[b] = mean_g(reward[b,g])
std[b] = sqrt(mean_g((reward[b,g]-mean[b])²))
advantage[b,g] = (reward[b,g]-mean[b])/(std[b]+1e-8)
```

同一个 response 的 advantage 会广播到全部 S 个 token，随后计算：

```text
ratio = exp(log_pi-log_pi_old)
surrogate = min(ratio×advantage, clamp(ratio,1-eps,1+eps)×advantage)
d = log_ref-log_pi
kl = exp(d)-d-1
output[0] = -mean(surrogate-beta×kl)
```

最终均值覆盖完整的 `B×G×S`，不使用 mask。

### 实现要求

- 保持 `solve` 签名不变，只使用原生功能
- 使用 population standard deviation，即除以 G
- 正确保留 KL 差值方向、beta 符号和最终整体负号

### 示例

```text
B=1，G=2，S=2，rewards=[10,0]
clip_eps=0.2，beta=0.01
output[0]≈-0.17563
```

### 约束

- `1≤B≤256`，`2≤G≤32`，`1≤S≤16384`
- `0≤clip_eps<1`，`0≤beta≤1`
- 两类 log 差值均位于 `[-16,16]`
- 性能测试为 `B=64,G=16,S=4096`
- 绝对误差和相对误差均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// rewards, log_pi, log_pi_old, log_ref, output are device pointers
extern "C" void solve(
    const float* rewards, const float* log_pi,
    const float* log_pi_old, const float* log_ref,
    float* output, float clip_eps, float beta,
    int B, int G, int S
) {}
```

starter 没有提供 advantage 缓冲区，也没有给出归约实现。S 个位置全部参与计算。

## 实现思路

第一阶段中，每个线程负责一个 `(b,g)`，按照 g 的固定顺序计算该 prompt 的均值和总体方差，再得到 advantage。同一个 b 的统计量会由 G 个线程重复计算，但 G≤32，这样可以省去跨 kernel 的统计同步。

第二阶段逐 token 依次计算 ratio、clip、surrogate、`d=log_ref-log_pi` 和 `surrogate-beta*kl`，再在 Block 内按固定顺序归约。最终 kernel 按 partial 的顺序累加，添加整体负号并除以 `BGS`。整个过程不使用原子加或显式 FMA，从而固定归约顺序，并保留原公式的运算边界。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_advantages(const float* rewards, float* advantages, int B, int G) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= B * G) return;
    int b = index / G;
    float mean = 0.0f;
    for (int g = 0; g < G; ++g) mean += rewards[b * G + g];
    mean /= static_cast<float>(G);
    float variance = 0.0f;
    for (int g = 0; g < G; ++g) {
        float difference = rewards[b * G + g] - mean;
        variance += difference * difference;
    }
    variance /= static_cast<float>(G);
    float standardDeviation = sqrtf(variance);
    advantages[index] = (rewards[index] - mean) / (standardDeviation + 1.0e-8f);
}

__global__ void grpo_partial(
    const float* advantages, const float* log_pi,
    const float* log_pi_old, const float* log_ref,
    float* partial, float clip_eps, float beta,
    int G, int S, int count
) {
    __shared__ float sums[256];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float objective = 0.0f;
    if (index < count) {
        int response = index / S;
        float advantage = advantages[response];
        float ratio = expf(log_pi[index] - log_pi_old[index]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float plain = ratio * advantage;
        float bounded = clipped * advantage;
        float surrogate = fminf(plain, bounded);
        float difference = log_ref[index] - log_pi[index];
        float kl = expf(difference) - difference - 1.0f;
        objective = surrogate - beta * kl;
    }
    sums[threadIdx.x] = objective;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}

__global__ void grpo_finish(const float* partial, float* output, int blocks, int count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blocks; ++i) sum += partial[i];
        output[0] = -sum / static_cast<float>(count);
    }
}

extern "C" void solve(
    const float* rewards, const float* log_pi,
    const float* log_pi_old, const float* log_ref,
    float* output, float clip_eps, float beta,
    int B, int G, int S
) {
    constexpr int threads = 256;
    int responses = B * G;
    int count = responses * S;
    int blocks = (count + threads - 1) / threads;
    float *advantages = nullptr, *partial = nullptr;
    cudaMalloc(&advantages, static_cast<size_t>(responses) * sizeof(float));
    cudaMalloc(&partial, static_cast<size_t>(blocks) * sizeof(float));
    compute_advantages<<<(responses + threads - 1) / threads, threads>>>(rewards, advantages, B, G);
    grpo_partial<<<blocks, threads>>>(advantages, log_pi, log_pi_old, log_ref,
                                     partial, clip_eps, beta, G, S, count);
    grpo_finish<<<1, 1>>>(partial, output, blocks, count);
    cudaDeviceSynchronize();
    cudaFree(partial);
    cudaFree(advantages);
}
```
