---
id: ppo-clipped-surrogate-loss
leetgpuId: 107
titleEn: PPO Clipped Surrogate Loss
titleZh: PPO 截断代理损失
description: 使用 CUDA 计算逐 token PPO 概率比、截断代理目标及跨批次与序列的标量均值损失
problemUrl: https://leetgpu.com/challenges/ppo-clipped-surrogate-loss
updatedAt: 2026-11-12
---
## 题目定义

输入 `advantages`、`log_pi` 和 `log_pi_old` 的形状均为 `[B,S]`，数据类型为 FP32。`S` 个位置全部是有效 token，不使用 mask。逐元素计算：

```text
ratio = exp(log_pi - log_pi_old)
clipped = clamp(ratio, 1-clip_eps, 1+clip_eps)
surrogate = min(ratio×advantage, clipped×advantage)
output[0] = -mean(surrogate)
```

`min` 对正、负 advantage 都按同一公式取值，不能根据符号固定选择某个截断项。最后的负号把需要最大化的代理目标转换成需要最小化的损失。

### 实现要求

- 保持 `solve` 签名不变
- 只使用原生功能
- 在 `output[0]` 写入跨全部 `B×S` 元素的单个 FP32 均值

### 示例

```text
B=1，S=4，clip_eps=0.2
advantages = [1, -2, 3, -4]
ratio = [1.3, 0.7, 1.1, 0.8]
clipped = [1.2, 0.8, 1.1, 0.8]
surrogate = [1.2, -1.6, 3.3, -3.2]
output[0] = 0.075
```

### 约束

- `1 ≤ B ≤ 256`，`1 ≤ S ≤ 16,384`
- `0 ≤ clip_eps < 1`
- `log_pi-log_pi_old ∈ [-16,16]`
- 性能测试为 `B=256`、`S=16,384`
- 绝对误差和相对误差均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(
    const float* advantages,
    const float* log_pi,
    const float* log_pi_old,
    float* output,
    float clip_eps,
    int B,
    int S
) {}
```

输入已经给出同一 token 的 advantage 和两套 log probability，因此不需要 Softmax、token 筛选或序列 mask。starter 没有提供 kernel 或工作区。

## 实现思路

第一阶段由每个线程按原公式计算一个 surrogate。为避免改变舍入顺序，乘法不与其他表达式合并成 FMA，正负 advantage 的截断逻辑也保持原样。Block 内按照固定的二叉树顺序，归约出一个 FP32 partial sum。

第二阶段由单线程按照 partial 数组的顺序累加，最后执行 `-sum/(B*S)`。固定的归约顺序避开了 `atomicAdd` 带来的不确定性；partial 数量很小，所有元素仍会计入最终均值。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void ppo_partial(
    const float* advantages,
    const float* log_pi,
    const float* log_pi_old,
    float* partial,
    float clip_eps,
    int count
) {
    __shared__ float sums[256];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    if (index < count) {
        float ratio = expf(log_pi[index] - log_pi_old[index]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float plain = ratio * advantages[index];
        float bounded = clipped * advantages[index];
        value = fminf(plain, bounded);
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}

__global__ void ppo_finish(const float* partial, float* output, int blocks, int count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blocks; ++i) sum += partial[i];
        output[0] = -sum / static_cast<float>(count);
    }
}

extern "C" void solve(
    const float* advantages,
    const float* log_pi,
    const float* log_pi_old,
    float* output,
    float clip_eps,
    int B,
    int S
) {
    int count = B * S;
    constexpr int threads = 256;
    int blocks = (count + threads - 1) / threads;
    float* partial = nullptr;
    cudaMalloc(&partial, static_cast<size_t>(blocks) * sizeof(float));
    ppo_partial<<<blocks, threads>>>(advantages, log_pi, log_pi_old, partial, clip_eps, count);
    ppo_finish<<<1, 1>>>(partial, output, blocks, count);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
```
