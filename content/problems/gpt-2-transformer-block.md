---
id: gpt-2-transformer-block
leetgpuId: 74
titleEn: GPT-2 Transformer Block
titleZh: GPT-2 Transformer 块
description: 按固定 GPT-2 124M 维度实现双 Pre-LayerNorm、全局多头注意力、tanh GELU、投影与两次残差连接
problemUrl: https://leetgpu.com/challenges/gpt-2-transformer-block
updatedAt: 2026-11-12
---
## 题目定义

实现一个固定维度的 GPT-2 Transformer Block：隐藏维度 `D=768`，注意力头数 `H=12`，每头维度 `DH=64`，FFN 中间维度 `3072`

运行时顺序为：

```text
x_norm = LayerNorm(x)
q,k,v = split(x_norm @ W_qkv + b_qkv)
attn = Softmax(q @ kᵀ / sqrt(64)) @ v
hidden = x + concat(attn_heads) @ W_attn + b_attn
h_norm = LayerNorm(hidden)
fc = GELU_tanh(h_norm @ W_fc + b_fc)
output = hidden + fc @ W_proj + b_proj
```

注意力是双向全局注意力，runtime reference 没有 causal mask。LayerNorm 使用总体方差和 `eps=1e-5`，GELU 使用 PyTorch 的 `approximate="tanh"`

### 官方示例

```text
输入：x.shape       = (4, 768)
      weights.shape = (7,087,872,)
      seq_len       = 4
输出：output.shape  = (4, 768)
```

示例中的 `x` 从 `[-1,1]` 均匀抽样，`weights` 按下述固定布局随机初始化

### 完整约束

- `1 ≤ seq_len ≤ 4096`
- `x` 和 `output` 的形状均为 `(seq_len,768)`
- `weights` 是恰含 `7,087,872` 个元素的连续缓冲区
- `x`、`output` 和 `weights` 均为 FP32
- 固定架构为 `d_model=768`、`n_heads=12`、`head_dim=64`、`ffn_dim=3072`
- LayerNorm 使用 `eps=1e-5`，GELU 使用 tanh 近似
- 性能测试使用 `seq_len=1024`
- 功能测试覆盖 `seq_len=1,2,4,16,30,64,100,128,256` 及零输入
- 判定容差为 `atol=0.001, rtol=0.001`

## 模板代码解析

真实 starter 为：

```cpp
#include <cuda_runtime.h>

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output,
                      const float* weights, int seq_len) {}
```

`weights` 是一个连续 FP32 缓冲区，布局严格为：

```text
ln1_w[768], ln1_b[768]
W_qkv[768,2304], b_qkv[2304]
W_attn[768,768], b_attn[768]
ln2_w[768], ln2_b[768]
W_fc[768,3072], b_fc[3072]
W_proj[3072,768], b_proj[768]
```

所有二维权重都是行主序，投影语义是输入行向量右乘权重，也就是输出列 `j` 累加 `input[k] * W[k,out]`

## 实现思路

参考实现把各阶段拆为顺序 kernel，并复用工作区以控制内存峰值：`a` 保存归一化/投影结果，`qkv` 保存 2304 维投影，`attention` 保存拼接后的注意力结果，`hidden` 保存第一次残差，`fc` 保存 3072 维 FFN 激活

LayerNorm 为每个 token 分配一个线程，按固定维度计算均值与总体方差。注意力为每个 `(head,query)` 启动一个 Block：线程 0 顺序计算所有 Key 的 score 和减最大值 Softmax，再由线程协作输出该头的 64 个特征。这里没有上三角掩码

QKV 在每个 token 内布局为 `[Q的768维,K的768维,V的768维]`，每个部分再按 `head*64+feature` 定位。注意力缩放固定为 `1/sqrt(64)=0.125`

两次残差不可交换：第一次把注意力投影加到原始 `x`，得到 `hidden`；第二次把 FFN 投影加到 `hidden`。LayerNorm 2 的输入必须是 `hidden`

`tanh` GELU 公式为：

```text
0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))
```

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr int D = 768;
constexpr int H = 12;
constexpr int DH = 64;
constexpr int FFN = 3072;
constexpr int O_LN1_W = 0;
constexpr int O_LN1_B = O_LN1_W + D;
constexpr int O_WQKV = O_LN1_B + D;
constexpr int O_BQKV = O_WQKV + D * 3 * D;
constexpr int O_WAPROJ = O_BQKV + 3 * D;
constexpr int O_BAPROJ = O_WAPROJ + D * D;
constexpr int O_LN2_W = O_BAPROJ + D;
constexpr int O_LN2_B = O_LN2_W + D;
constexpr int O_WFC = O_LN2_B + D;
constexpr int O_BFC = O_WFC + D * FFN;
constexpr int O_WPROJ = O_BFC + FFN;
constexpr int O_BPROJ = O_WPROJ + FFN * D;

__global__ void layer_norm(const float* input, float* output,
                           const float* weight, const float* bias,
                           int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float mean = 0.0f;
    for (int k = 0; k < D; ++k) mean += input[row * D + k];
    mean /= D;
    float variance = 0.0f;
    for (int k = 0; k < D; ++k) {
        float diff = input[row * D + k] - mean;
        variance += diff * diff;
    }
    variance /= D;
    float inverse = rsqrtf(variance + 1e-5f);
    for (int k = 0; k < D; ++k)
        output[row * D + k] = (input[row * D + k] - mean)
                              * inverse * weight[k] + bias[k];
}

__global__ void linear(const float* input, float* output,
                       const float* weight, const float* bias,
                       int rows, int input_dim, int output_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * output_dim;
    if (index >= total) return;
    int row = index / output_dim;
    int out = index % output_dim;
    float sum = bias[out];
    for (int k = 0; k < input_dim; ++k)
        sum += input[row * input_dim + k] * weight[k * output_dim + out];
    output[index] = sum;
}

__global__ void attention_kernel(const float* qkv, float* attention,
                                 int seq_len) {
    extern __shared__ float probabilities[];
    int query = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    if (tid == 0) {
        float maximum = -CUDART_INF_F;
        for (int key = 0; key < seq_len; ++key) {
            float score = 0.0f;
            for (int k = 0; k < DH; ++k) {
                float q = qkv[query * 3 * D + head * DH + k];
                float kval = qkv[key * 3 * D + D + head * DH + k];
                score += q * kval;
            }
            score *= 0.125f;
            probabilities[key] = score;
            maximum = fmaxf(maximum, score);
        }
        float denominator = 0.0f;
        for (int key = 0; key < seq_len; ++key) {
            probabilities[key] = expf(probabilities[key] - maximum);
            denominator += probabilities[key];
        }
        for (int key = 0; key < seq_len; ++key)
            probabilities[key] /= denominator;
    }
    __syncthreads();

    for (int feature = tid; feature < DH; feature += blockDim.x) {
        float sum = 0.0f;
        for (int key = 0; key < seq_len; ++key) {
            float value = qkv[key * 3 * D + 2 * D + head * DH + feature];
            sum += probabilities[key] * value;
        }
        attention[query * D + head * DH + feature] = sum;
    }
}

__global__ void add_residual(const float* residual, float* values, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] += residual[index];
}

__global__ void gelu_tanh(float* values, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float x = values[index];
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    values[index] = 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" void solve(const float* x, float* output,
                      const float* weights, int seq_len) {
    float *a = nullptr, *qkv = nullptr, *attention = nullptr;
    float *hidden = nullptr, *fc = nullptr;
    cudaMalloc(&a, static_cast<size_t>(seq_len) * D * sizeof(float));
    cudaMalloc(&qkv, static_cast<size_t>(seq_len) * 3 * D * sizeof(float));
    cudaMalloc(&attention, static_cast<size_t>(seq_len) * D * sizeof(float));
    cudaMalloc(&hidden, static_cast<size_t>(seq_len) * D * sizeof(float));
    cudaMalloc(&fc, static_cast<size_t>(seq_len) * FFN * sizeof(float));

    constexpr int threads = 256;
    int rows = (seq_len + threads - 1) / threads;
    layer_norm<<<rows, threads>>>(x, a, weights + O_LN1_W,
                                  weights + O_LN1_B, seq_len);
    int qkv_count = seq_len * 3 * D;
    linear<<<(qkv_count + threads - 1) / threads, threads>>>(
        a, qkv, weights + O_WQKV, weights + O_BQKV, seq_len, D, 3 * D);
    dim3 attention_grid(seq_len, H);
    attention_kernel<<<attention_grid, 64,
                       static_cast<size_t>(seq_len) * sizeof(float)>>>(
        qkv, attention, seq_len);
    int hidden_count = seq_len * D;
    linear<<<(hidden_count + threads - 1) / threads, threads>>>(
        attention, hidden, weights + O_WAPROJ, weights + O_BAPROJ,
        seq_len, D, D);
    add_residual<<<(hidden_count + threads - 1) / threads, threads>>>(
        x, hidden, hidden_count);
    layer_norm<<<rows, threads>>>(hidden, a, weights + O_LN2_W,
                                  weights + O_LN2_B, seq_len);
    int fc_count = seq_len * FFN;
    linear<<<(fc_count + threads - 1) / threads, threads>>>(
        a, fc, weights + O_WFC, weights + O_BFC, seq_len, D, FFN);
    gelu_tanh<<<(fc_count + threads - 1) / threads, threads>>>(fc, fc_count);
    linear<<<(hidden_count + threads - 1) / threads, threads>>>(
        fc, output, weights + O_WPROJ, weights + O_BPROJ,
        seq_len, FFN, D);
    add_residual<<<(hidden_count + threads - 1) / threads, threads>>>(
        hidden, output, hidden_count);
    cudaDeviceSynchronize();

    cudaFree(fc);
    cudaFree(hidden);
    cudaFree(attention);
    cudaFree(qkv);
    cudaFree(a);
}
```
