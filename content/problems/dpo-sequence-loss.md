---
id: dpo-sequence-loss
leetgpuId: 108
titleEn: DPO Sequence Loss
titleZh: DPO 序列损失
description: 使用 CUDA 从当前策略与参考策略的序列 log probability 计算数值稳定的 DPO 标量均值损失
problemUrl: https://leetgpu.com/challenges/dpo-sequence-loss
updatedAt: 2026-11-12
---
## 题目定义

四个输入都是长度 `B` 的 FP32 序列 log probability。token 级 log probability 已按照相同的 tokenization 和 mask 规则求和，本题无需再次做 token 归约。对每个样本：

```text
current_margin = chosen_logps - rejected_logps
reference_margin = chosen_ref_logps - rejected_ref_logps
z = beta × (current_margin - reference_margin)
loss = softplus(-z) = -log(sigmoid(z))
output[0] = mean(loss)
```

计算时必须用当前策略 margin 减去参考策略 margin，最终损失不带额外负号。

### 实现要求

- 保持 `solve` 签名不变
- 只使用原生功能
- 稳定计算 Softplus，并将 `B` 个损失的均值写入 `output[0]`

### 示例

```text
B=4，beta=0.1
chosen=[0,1,-1,2]，其余三个输入全0
z=[0,0.1,-0.1,0.2]
output[0]≈0.67002
```

### 约束

- `1 ≤ B ≤ 65,536`
- `0 < beta ≤ 1`
- 所有输入有限，且均为已经求和的序列 log probability
- 性能测试使用 `B=65,536`
- 绝对误差和相对误差均为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
extern "C" void solve(
    const float* chosen_logps,
    const float* rejected_logps,
    const float* chosen_ref_logps,
    const float* rejected_ref_logps,
    float* output,
    float beta,
    int B
) {}
```

starter 接收四个设备数组和标量 `beta`，但没有提供归约 kernel。四个数组已经是序列级数据，不应再除以序列长度，也不应应用 mask。

## 实现思路

两个 margin 必须严格按照 reference 的括号顺序计算，再相减并乘 `beta`。四项不能重排成另一种加减顺序，也不能合并成 FMA，以免大负 log probability 相减时改变舍入结果。

Softplus 使用稳定公式：

```text
softplus(x) = max(x,0) + log1p(exp(-abs(x)))
```

即使输入 log probability 达到 `±1000/±500`，经 `beta=0.1` 形成 `±100/±50` 的 DPO logit，这种写法也不会溢出。Block 内按固定顺序归约，第二个 kernel 再依照 partial 顺序累加并除以 `B`。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void dpo_partial(
    const float* chosen,
    const float* rejected,
    const float* chosenRef,
    const float* rejectedRef,
    float* partial,
    float beta,
    int B
) {
    __shared__ float sums[256];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float loss = 0.0f;
    if (index < B) {
        float currentMargin = chosen[index] - rejected[index];
        float referenceMargin = chosenRef[index] - rejectedRef[index];
        float z = beta * (currentMargin - referenceMargin);
        float x = -z;
        loss = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    }
    sums[threadIdx.x] = loss;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}

__global__ void dpo_finish(const float* partial, float* output, int blocks, int B) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blocks; ++i) sum += partial[i];
        output[0] = sum / static_cast<float>(B);
    }
}

extern "C" void solve(
    const float* chosen_logps,
    const float* rejected_logps,
    const float* chosen_ref_logps,
    const float* rejected_ref_logps,
    float* output,
    float beta,
    int B
) {
    constexpr int threads = 256;
    int blocks = (B + threads - 1) / threads;
    float* partial = nullptr;
    cudaMalloc(&partial, static_cast<size_t>(blocks) * sizeof(float));
    dpo_partial<<<blocks, threads>>>(chosen_logps, rejected_logps, chosen_ref_logps,
                                    rejected_ref_logps, partial, beta, B);
    dpo_finish<<<1, 1>>>(partial, output, blocks, B);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
```
