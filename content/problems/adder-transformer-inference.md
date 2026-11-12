---
id: adder-transformer-inference
leetgpuId: 76
titleEn: Adder Transformer Inference
titleZh: 加法器 Transformer 推理
description: 精确复现手工构造的十参数加法器 Transformer，并自回归生成 11 位输出 logits
problemUrl: https://leetgpu.com/challenges/adder-transformer-inference
updatedAt: 2026-11-12
---
## 题目定义

输入 `prompts` 的形状为 `[batch_size,31]`，包含两个倒序排列的十进制十位数；模型自回归生成 11 步，每步输出 10 个 token 的 logits，因此 `output` 形状为 `[batch_size,11,10]`

模型维度固定为 2，单头维度为 2，词表大小为 10，权重缓冲仅有 10 个 FP32 参数：

```text
embed_w[2], q_w[2], v_w[1], gate_w[2], carry_w[1], norm_w[2]
```

每一步都对当前完整序列重新执行模型，只取最后一个有效位置的 logits 写入输出，再以 `argmax` token 追加到序列

### 官方示例

```text
输入 prompts（shape [2,31]）：
  [0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0]
  [0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
输出 logits shape：[2,11,10]
argmax 解码：
  (3,5)  → [8,0,0,0,0,0,0,0,0,0,0]
  (99,1) → [0,0,1,0,0,0,0,0,0,0,0]
```

### 完整约束

- `1 ≤ batch_size ≤ 100000`
- `prompts` 的形状为 `[batch_size,31]`，数据类型为 `int32`，元素范围为 `[0,9]`
- `weights` 是恰含 10 个元素的 FP32 数组
- `output` 的形状为 `[batch_size,11,10]`，数据类型为 FP32
- 输入加数范围为 `[0,9,999,999,999]`
- 固定架构为词表 10、隐藏维度 2、单头、头维度 2、prompt 长度 31、解码 11 步
- RMSNorm epsilon 为 `1e-6`，RoPE 角频率为 `2π/19`
- 性能测试使用 `batch_size=100000`
- 判定容差为 `atol=0.01, rtol=0.01`

## 模板代码解析

真实 CUDA 入口为：

```cpp
#include <cuda_runtime.h>

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output,
                      const float* weights, int batch_size) {}
```

运行时语义必须逐项保持：embedding 为 `[w0-w1*d²,-d]`；注意力前和 Q/K 投影后都用无参数 RMSNorm，epsilon 为 `1e-6`；Q/K 应用周期 19 的 RoPE；注意力是 causal Softmax；O 投影把注意力第一维放到隐藏第二维；MLP 使用两个特定 gate 与 SwiGLU 差值；最后使用带学习权重的 RMSNorm，并与 embedding table 权重绑定做输出投影

由于每一步只需要最后位置的 logits，而 causal mask 保证它可以关注当前的全部 token，因此参考 CUDA 实现只计算最后一个 Query 对所有 Key 的注意力，无需计算其他 Query 的输出。这与 Python runtime 在最后位置的结果相同

## 实现思路

一个 CUDA 线程处理一个 batch 样本，每个 Block 使用 256 个线程，并行处理最多 256 个样本。每个线程顺序执行 11 步完整的最后位置推理，并在自己的局部数组中维护最长 42 个 token，不需要跨线程通信或全局临时缓冲

每个序列 token 先由 embedding 构造二维 `h`，再做单位 RMSNorm。Q 为 `[h0*qw0,h0*qw1]`，K 为 `[h0,0]`，V 为 `[h1*vw,0]`；Q/K 再各自执行 RMSNorm，然后按位置角度 `position*2π/19` 旋转

最后一个 Query 的注意力 score，是旋转后的 Q/K 点积乘以 `ATTN_SCALE = (HEAD_DIM^-0.5) × QK_NORM_SCALE²`。运行时该值为 `52.91764988494996`，写入 CUDA FP32 常量后是 `52.91764831542969f`（位模式 `0x4253abac`）。Q/K 的 unit RMSNorm 不包含 `QK_NORM_SCALE`，这个放大系数只能通过 `ATTN_SCALE` 保留。Softmax 先减去最大值，再对当前 `seq_len` 个 Key 全部归一化，这等价于 causal mask 在最后一行的行为

注意力输出的第一维形成 `o=[0,attn0]`，并与最后一个 token 的原始 embedding 相加。第二次 RMSNorm 后，严格复现：

```text
g0 = h0*a + h1*c
g1 = h0*(a-c/1000) + h1*c
mix = silu(gate) * h0
mlp = [0, carry*(mix1-mix0)]
```

加上 MLP 残差后执行学习型 RMSNorm，最后与十个 embedding 向量做点积，得到 logits。`argmax` 使用严格 `>`，因此平局时保留最小 token 下标，与 PyTorch `argmax` 一致

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr int VOCAB = 10;
constexpr int PROMPT = 31;
constexpr int STEPS = 11;
constexpr int MAX_LEN = PROMPT + STEPS;
constexpr float RMS_EPS = 1e-6f;
constexpr float PI = 3.14159265358979323846f;
constexpr float ATTN_SCALE = 52.91764831542969f;

__device__ void normalize2(float& x, float& y) {
    float inverse = rsqrtf(0.5f * (x * x + y * y) + RMS_EPS);
    x *= inverse;
    y *= inverse;
}

__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void adder_inference(const int* prompts, float* output,
                                const float* weights, int batch_size) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batch_size) return;

    int sequence[MAX_LEN];
    for (int i = 0; i < PROMPT; ++i)
        sequence[i] = prompts[batch * PROMPT + i];

    float embed0 = weights[0], embed1 = weights[1];
    float qw0 = weights[2], qw1 = weights[3];
    float vw = weights[4];
    float gate0 = weights[5], gate1 = weights[6];
    float carry = weights[7];
    float norm0 = weights[8], norm1 = weights[9];
    float omega = 2.0f * PI / 19.0f;

    for (int step = 0; step < STEPS; ++step) {
        int length = PROMPT + step;
        int last = length - 1;
        int queryToken = sequence[last];
        float hq0 = embed0 - embed1 * queryToken * queryToken;
        float hq1 = -static_cast<float>(queryToken);
        float original0 = hq0, original1 = hq1;
        normalize2(hq0, hq1);

        float q0 = hq0 * qw0;
        float q1 = hq0 * qw1;
        normalize2(q0, q1);
        float qAngle = last * omega;
        float qCos = cosf(qAngle), qSin = sinf(qAngle);
        float qr0 = q0 * qCos - q1 * qSin;
        float qr1 = q0 * qSin + q1 * qCos;

        float maximum = -CUDART_INF_F;
        float scores[MAX_LEN];
        for (int pos = 0; pos < length; ++pos) {
            int token = sequence[pos];
            float k0 = embed0 - embed1 * token * token;
            float k1 = -static_cast<float>(token);
            normalize2(k0, k1);
            k1 = 0.0f;
            normalize2(k0, k1);
            float angle = pos * omega;
            float cosine = cosf(angle), sine = sinf(angle);
            float kr0 = k0 * cosine - k1 * sine;
            float kr1 = k0 * sine + k1 * cosine;
            scores[pos] = (qr0 * kr0 + qr1 * kr1) * ATTN_SCALE;
            maximum = fmaxf(maximum, scores[pos]);
        }

        float denominator = 0.0f;
        for (int pos = 0; pos < length; ++pos) {
            scores[pos] = expf(scores[pos] - maximum);
            denominator += scores[pos];
        }
        float attention0 = 0.0f;
        for (int pos = 0; pos < length; ++pos) {
            int token = sequence[pos];
            float v0 = embed0 - embed1 * token * token;
            float v1 = -static_cast<float>(token);
            normalize2(v0, v1);
            v0 = v1 * vw;
            attention0 += (scores[pos] / denominator) * v0;
        }

        float h0 = original0;
        float h1 = original1 + attention0;
        float n0 = h0, n1 = h1;
        normalize2(n0, n1);
        float g0 = n0 * gate0 + n1 * gate1;
        float g1 = n0 * (gate0 - gate1 / 1000.0f) + n1 * gate1;
        float mix0 = silu(g0) * n0;
        float mix1 = silu(g1) * n0;
        h1 += carry * (mix1 - mix0);

        normalize2(h0, h1);
        h0 *= norm0;
        h1 *= norm1;

        int bestToken = 0;
        float bestLogit = -CUDART_INF_F;
        for (int token = 0; token < VOCAB; ++token) {
            float e0 = embed0 - embed1 * token * token;
            float e1 = -static_cast<float>(token);
            float logit = h0 * e0 + h1 * e1;
            output[(batch * STEPS + step) * VOCAB + token] = logit;
            if (logit > bestLogit) {
                bestLogit = logit;
                bestToken = token;
            }
        }
        sequence[length] = bestToken;
    }
}

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output,
                      const float* weights, int batch_size) {
    constexpr int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    adder_inference<<<blocks, threads>>>(prompts, output, weights, batch_size);
    cudaDeviceSynchronize();
}
```
