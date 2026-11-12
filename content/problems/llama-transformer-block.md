---
id: llama-transformer-block
leetgpuId: 93
titleEn: Llama Transformer Block
titleZh: Llama Transformer 块
description: 复现固定维度 Llama 块的 RMSNorm、RoPE、因果 GQA、SwiGLU 与两次残差连接
problemUrl: https://leetgpu.com/challenges/llama-transformer-block
updatedAt: 2026-11-12
---
## 题目定义

实现一个固定架构的 Llama Transformer Block：`D=512`、8 个 Query 头、2 个 KV 头、`head_dim=64`、SwiGLU 中间维度 1408。runtime 按以下顺序执行：

1. `RMSNorm(x,w1,eps=1e-5)`，再做 Q/K/V 投影。
2. 将每个头拆成前后各 32 维，使用 `[seq_len,32]` 的 cos/sin 对 Q、K 应用 RoPE。
3. 每 4 个连续 Q 头共享一个 KV 头，执行 `1/sqrt(64)` 缩放的 causal attention。
4. O 投影后加原始 `x`，得到第一次残差 `hidden`。
5. `RMSNorm(hidden,w2)`，计算 `silu(W_gate*h) * (W_up*h)`，经 `W_down` 投影后加 `hidden`。

### 官方示例

```text
seq_len=4
x.shape=(4,512)
weights.shape=(2,819,072,)
cos.shape=sin.shape=(4,32)
output.shape=(4,512)
```

### 完整约束与权重布局

- `1 ≤ seq_len ≤ 4096`，所有张量为 FP32；性能测试使用 `seq_len=2048`
- 所有二维权重均为 `(out_dim,in_dim)` 行主序，无 bias
- `w1[512] @0`；`WQ[512,512] @512`；`WK[128,512] @262656`
- `WV[128,512] @328192`；`WO[512,512] @393728`；`w2[512] @655872`
- `Wgate[1408,512] @656384`；`Wup[1408,512] @1377280`
- `Wdown[512,1408] @2098176`；总计 `2,819,072` 个 FP32
- 判定容差为 `1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// x, output, weights, cos, sin are device pointers
extern "C" void solve(const float* x, float* output, const float* weights,
                      const float* cos, const float* sin, int seq_len) {}
```

平台只提供一个 packed 权重缓冲区；参考实现根据上述 offset 取得各段指针，并自行分配 Q/K/V、注意力和 FFN 中间缓冲。

## 实现思路

参考实现把计算拆到多个 Grid 并行 kernel 中，避免由单个线程串行完成整个 Block。矩阵投影由每个线程计算一个输出元素；注意力由一个 Block 处理 `(q_head,query)`，只遍历 `key≤query`，并把最多 4096 个权重存入 Shared Memory；RoPE、残差、SiLU 和乘法则交给逐元素 kernel。

RoPE 必须采用 half-split 形式：`[a,b] -> [a*cos-b*sin,a*sin+b*cos]`，而不是相邻偶奇配对。GQA 映射为 `kv_head=q_head/4`；两个 Pre-RMSNorm 和两次残差也要严格遵循 runtime 的执行顺序。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr int D = 512, QH = 8, KVH = 2, HD = 64, F = 1408;
constexpr int O_WQ = 512, O_WK = 262656, O_WV = 328192, O_WO = 393728;
constexpr int O_RMS2 = 655872, O_GATE = 656384, O_UP = 1377280, O_DOWN = 2098176;

__global__ void rms(const float *x, const float *w, float *y, int rows) {
  int r = blockIdx.x;
  if (r >= rows)
    return;
  if (threadIdx.x == 0) {
    float s = 0;
    for (int j = 0; j < D; ++j)
      s += x[r * D + j] * x[r * D + j];
    float a = rsqrtf(s / D + 1e-5f);
    for (int j = 0; j < D; ++j)
      y[r * D + j] = x[r * D + j] * a * w[j];
  }
}
__global__ void linear(const float *x, const float *w, float *y, int rows, int in, int out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y;
  if (r < rows && o < out) {
    float s = 0;
    for (int j = 0; j < in; ++j)
      s += x[r * in + j] * w[o * in + j];
    y[r * out + o] = s;
  }
}
__global__ void rope(float *q, float *k, const float *c, const float *s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * (QH + KVH) * 32;
  if (i >= total)
    return;
  int half = i % 32;
  int z = i / 32;
  int h = z % (QH + KVH);
  int t = z / (QH + KVH);
  float *a = h < QH ? q + (t * QH + h) * HD : k + (t * KVH + h - QH) * HD;
  float x = a[half], y = a[half + 32], co = c[t * 32 + half], si = s[t * 32 + half];
  a[half] = x * co - y * si;
  a[half + 32] = x * si + y * co;
}
__global__ void attention(const float *q, const float *k, const float *v, float *y) {
  extern __shared__ float p[];
  int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, kh = h / 4;
  if (tid == 0) {
    float m = -CUDART_INF_F;
    for (int u = 0; u <= t; ++u) {
      float z = 0;
      for (int j = 0; j < HD; ++j)
        z += q[(t * QH + h) * HD + j] * k[(u * KVH + kh) * HD + j];
      p[u] = z * 0.125f;
      m = fmaxf(m, p[u]);
    }
    float d = 0;
    for (int u = 0; u <= t; ++u) {
      p[u] = expf(p[u] - m);
      d += p[u];
    }
    for (int u = 0; u <= t; ++u)
      p[u] /= d;
  }
  __syncthreads();
  for (int j = tid; j < HD; j += blockDim.x) {
    float z = 0;
    for (int u = 0; u <= t; ++u)
      z += p[u] * v[(u * KVH + kh) * HD + j];
    y[(t * D + h * HD) + j] = z;
  }
}
__global__ void add(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] + b[i];
}
__global__ void swiglu(float *g, const float *u, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = g[i];
    g[i] = (x / (1.0f + expf(-x))) * u[i];
  }
}
extern "C" void solve(const float *x, float *output, const float *w, const float *c, const float *s,
                      int n) {
  float *a, *q, *k, *v, *att, *hid, *g, *u, *ff;
  cudaMalloc(&a, (size_t)n * D * 4);
  cudaMalloc(&q, (size_t)n * D * 4);
  cudaMalloc(&k, (size_t)n * 128 * 4);
  cudaMalloc(&v, (size_t)n * 128 * 4);
  cudaMalloc(&att, (size_t)n * D * 4);
  cudaMalloc(&hid, (size_t)n * D * 4);
  cudaMalloc(&g, (size_t)n * F * 4);
  cudaMalloc(&u, (size_t)n * F * 4);
  cudaMalloc(&ff, (size_t)n * D * 4);
  rms<<<n, 1>>>(x, w, a, n);
  dim3 b(256);
  linear<<<dim3((D + 255) / 256, n), b>>>(a, w + O_WQ, q, n, D, D);
  linear<<<dim3(1, n), b>>>(a, w + O_WK, k, n, D, 128);
  linear<<<dim3(1, n), b>>>(a, w + O_WV, v, n, D, 128);
  rope<<<(n * (QH + KVH) * 32 + 255) / 256, 256>>>(q, k, c, s, n);
  attention<<<dim3(n, QH), 256, (size_t)n * 4>>>(q, k, v, att);
  linear<<<dim3(2, n), b>>>(att, w + O_WO, a, n, D, D);
  add<<<(n * D + 255) / 256, 256>>>(x, a, hid, n * D);
  rms<<<n, 1>>>(hid, w + O_RMS2, a, n);
  linear<<<dim3((F + 255) / 256, n), b>>>(a, w + O_GATE, g, n, D, F);
  linear<<<dim3((F + 255) / 256, n), b>>>(a, w + O_UP, u, n, D, F);
  swiglu<<<(n * F + 255) / 256, 256>>>(g, u, n * F);
  linear<<<dim3(2, n), b>>>(g, w + O_DOWN, ff, n, F, D);
  add<<<(n * D + 255) / 256, 256>>>(hid, ff, output, n * D);
  cudaDeviceSynchronize();
  cudaFree(a);
  cudaFree(q);
  cudaFree(k);
  cudaFree(v);
  cudaFree(att);
  cudaFree(hid);
  cudaFree(g);
  cudaFree(u);
  cudaFree(ff);
}
```
