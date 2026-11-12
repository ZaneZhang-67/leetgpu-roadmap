---
id: top-p-sampling
leetgpuId: 60
titleEn: Top-p Sampling
titleZh: Top-p 采样
description: 对 logits 执行稳定 Softmax、确定性排序与 nucleus 截断，再按指定 PCG 哈希和指数竞赛采样
problemUrl: https://leetgpu.com/challenges/top-p-sampling
updatedAt: 2026-11-12
---
## 题目定义

给定长度为 `vocab_size` 的 FP32 `logits`、单元素阈值 `p` 和单元素 `int32` 种子 `seed`，执行 Top-p（nucleus）采样：

1. 对 logits 做 Softmax
2. 按概率降序排序；概率相等时按 token 下标升序
3. 选择累计概率首次达到或超过 `p` 的最短前缀
4. 在 nucleus 内重新归一化
5. 使用题目指定的 PCG 哈希与指数竞赛选出 token

每个 token `i` 的均匀随机数是：

```text
u_i = (h(h(uint32(seed)) + i) + 0.5) / 2^32
```

所有无符号整数运算按 `2^32` 回绕采样分数为 `q_i / (-ln(u_i))`，取分数最大的 nucleus token官方流程会显式重新归一化 nucleus 概率，参考实现保持这一步骤

### 实现要求与评测规模

- 不允许使用外部库
- `solve` 函数签名保持不变
- 输出写入单元素 `int32` 数组 `sampled_token`
- `3 ≤ vocab_size ≤ 50,000`，`-100 ≤ logits[i] ≤ 100`
- `0 < p ≤ 1`，输出 token 位于 `[0,vocab_size)`
- 功能测试覆盖词表大小 3、10、50、100、200、500、1000 和 5000
- 性能测试使用 `vocab_size=50000`、`p=0.9`、`seed=999`
- 排序必须保留等概率 token 的下标升序顺序
- nucleus 截断条件是累计概率 `>= p`，达到阈值的 token 也属于 nucleus

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* logits, const float* p,
                      const int* seed, int* sampled_token,
                      int vocab_size) {}
```

`logits`、`p`、`seed` 和 `sampled_token` 都是设备指针，阈值与种子必须从设备端读取，不能在 CPU 端直接解引用

参考实现使用设备端工作区保存概率和 token 下标，准备阶段按 FP32 顺序生成 Softmax 概率；排序阶段把长度补到不小于词表的最小 2 的幂，再通过并行 Bitonic 网络处理 `vocab_size=50000`，避免单线程平方复杂度排序

## 实现思路

先找最大 logit，再计算 `exp(logit-maxLogit)` 及其 FP32 总和，随后显式除以总和，得到官方实际排序的 FP32 概率；这样既避免指数溢出，也保留下溢概率归一化后可能出现的平局

Bitonic 工作长度 `L` 补到 2 的幂，填充项使用负概率；排序规则为概率降序，概率相等时原 token 下标升序；即使许多概率都下溢为零，结果仍与稳定降序排序一致；网络复杂度为 `O(L log²L)`，每一层由并行 kernel 完成

从排序首项开始做 FP32 累加，首个满足 `cumulative >= p[0]` 的位置也被保留；若浮点边界导致循环一直未触发，则保留整个词表；随后再次用 FP32 求 nucleus 概率和并逐项归一化

PCG 哈希使用 `uint32_t`，无符号乘加自然按 `2^32` 回绕；`seed[0]` 先按位解释为无符号数，再根据 token 原始下标生成 `u_i`；指数竞赛以 `double` 计算 `u`、对数和比值；遍历顺序与排序后的 nucleus 一致，并严格使用 `>` 更新，使完全相等的竞赛分数保留先出现的 token

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) {
    uint32_t state = x * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state)
                    * 277803737u;
    return (word >> 22u) ^ word;
}

__global__ void prepare_probabilities(const float* logits, float* values,
                                      int* indices, int vocab_size,
                                      int work_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float maxLogit = logits[0];
    for (int i = 1; i < vocab_size; ++i)
        maxLogit = fmaxf(maxLogit, logits[i]);

    float total = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        values[i] = expf(logits[i] - maxLogit);
        total += values[i];
        indices[i] = i;
    }
    for (int i = 0; i < vocab_size; ++i) values[i] /= total;
    for (int i = vocab_size; i < work_size; ++i) {
        values[i] = -1.0f;
        indices[i] = INT_MAX;
    }
}

__device__ __forceinline__ bool comes_before(float left_value, int left_index,
                                              float right_value, int right_index) {
    if (left_value > right_value) return true;
    if (left_value < right_value) return false;
    return left_index < right_index;
}

__global__ void bitonic_step(float* values, int* indices,
                             int work_size, int subsequence, int stride) {
    int left = blockIdx.x * blockDim.x + threadIdx.x;
    if (left >= work_size) return;
    int right = left ^ stride;
    if (right <= left) return;

    float left_value = values[left];
    float right_value = values[right];
    int left_index = indices[left];
    int right_index = indices[right];
    bool forward = (left & subsequence) == 0;
    bool swap = forward
        ? comes_before(right_value, right_index, left_value, left_index)
        : comes_before(left_value, left_index, right_value, right_index);
    if (swap) {
        values[left] = right_value;
        values[right] = left_value;
        indices[left] = right_index;
        indices[right] = left_index;
    }
}

__global__ void sample_nucleus(const float* values, const int* indices,
                               const float* p, const int* seed,
                               int* sampled_token, int vocab_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float cumulative = 0.0f;
    int nucleus_size = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        cumulative += values[i];
        if (cumulative >= p[0]) {
            nucleus_size = i + 1;
            break;
        }
    }

    float nucleus_total = 0.0f;
    for (int i = 0; i < nucleus_size; ++i) nucleus_total += values[i];

    uint32_t seed_hash = pcg_hash(static_cast<uint32_t>(seed[0]));
    double best_score = -1.0;
    int best_token = indices[0];
    for (int i = 0; i < nucleus_size; ++i) {
        int token = indices[i];
        uint32_t bits = pcg_hash(seed_hash + static_cast<uint32_t>(token));
        double uniform = (static_cast<double>(bits) + 0.5) / 4294967296.0;
        float probability = values[i] / nucleus_total;
        double score = static_cast<double>(probability) / -log(uniform);
        if (score > best_score) {
            best_score = score;
            best_token = token;
        }
    }
    sampled_token[0] = best_token;
}

extern "C" void solve(const float* logits, const float* p,
                      const int* seed, int* sampled_token,
                      int vocab_size) {
    int work_size = 1;
    while (work_size < vocab_size) work_size <<= 1;

    float* values = nullptr;
    int* indices = nullptr;
    cudaMalloc(&values, static_cast<size_t>(work_size) * sizeof(float));
    cudaMalloc(&indices, static_cast<size_t>(work_size) * sizeof(int));
    prepare_probabilities<<<1, 1>>>(logits, values, indices,
                                    vocab_size, work_size);

    constexpr int threads = 256;
    int blocks = (work_size + threads - 1) / threads;
    for (int subsequence = 2; subsequence <= work_size; subsequence <<= 1) {
        for (int stride = subsequence >> 1; stride > 0; stride >>= 1) {
            bitonic_step<<<blocks, threads>>>(values, indices, work_size,
                                               subsequence, stride);
        }
    }
    sample_nucleus<<<1, 1>>>(values, indices, p, seed,
                              sampled_token, vocab_size);
    cudaDeviceSynchronize();
    cudaFree(indices);
    cudaFree(values);
}
```
