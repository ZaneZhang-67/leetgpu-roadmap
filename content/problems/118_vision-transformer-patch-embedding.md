---
id: vision-transformer-patch-embedding
leetgpuId: 118
titleEn: Vision Transformer Patch Embedding
titleZh: Vision Transformer Patch Embedding
description: 使用 CUDA 将 NCHW 图像切分为 Patch、线性投影并加入 CLS 与位置嵌入
problemUrl: https://leetgpu.com/challenges/vision-transformer-patch-embedding
updatedAt: 2026-11-12
---
## 题目定义

输入 `images(B,C,H,W)`，按不重叠 `P×P` Patch 切分，行优先排列为 `N=(H/P)(W/P)` 个 Token。权重布局为 `patch_weight(D,C,P,P)`：

```text
token[b,n,d] = bias[d] + Σ images[b,c,py*P+i,px*P+j]*weight[d,c,i,j]
output[b,0,d] = cls_token[d] + pos_embed[0,d]
output[b,n+1,d] = token[b,n,d] + pos_embed[n+1,d]
```

CLS 不参与投影。

### 示例

官方 `1×1×4×4`、`P=2`、`D=2` 示例中，两个滤波器分别取每个 Patch 左上、右下像素，最终输出为 `[[0.5,-0.5],[2,5],[5,6],[12,11],[15,12]]`。

约束为 `B≤128`、`C≤4`、`H,W≤1024` 且可被 P 整除、`P≤32`、`D≤1024`，图像值位于 `[-10,10]`。性能用例为 `B=64,C=3,H=W=224,P=16,D=768`，绝对与相对容差均为 `1e-4`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* images, const float* patch_weight,
 const float* patch_bias, const float* cls_token, const float* pos_embed,
 float* output, int B, int C, int H, int W, int P, int D)
```

图像是 NCHW 连续布局；Patch 展平顺序为 C、patch row、patch col。输出序号 0 留给 CLS，Patch n 写入 n+1。

## 实现思路

每个线程负责一个 `(batch,patch,d)` 输出，并沿 `C×P×P` 累加。Patch、D、Batch 三维拥有充分并行度；性能用例会启动约 963 万个输出线程，不存在串行占位。权重对不同 Batch 与 Patch 复用，可进一步通过 Shared Memory tiled GEMM 优化；这里保持直接地址映射，避免物化 Patch 矩阵。

另一个 kernel 写 CLS 行。两者都直接加入位置嵌入。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void patch_kernel(const float* images, const float* weight, const float* bias,
                             const float* pos, float* output,
                             int B, int C, int H, int W, int P, int D, int N, int gw) {
    long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(B) * N * D;
    if (index >= total) return;
    int d = index % D;
    int n = (index / D) % N;
    int b = index / (static_cast<long long>(D) * N);
    int py = n / gw, px = n % gw;
    float sum = bias[d];
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < P; ++i)
            for (int j = 0; j < P; ++j) {
                int image_idx = ((b * C + c) * H + py * P + i) * W + px * P + j;
                int weight_idx = ((d * C + c) * P + i) * P + j;
                sum += images[image_idx] * weight[weight_idx];
            }
    output[(static_cast<long long>(b) * (N + 1) + n + 1) * D + d] = sum + pos[(n + 1) * D + d];
}

__global__ void cls_kernel(const float* cls, const float* pos, float* output, int B, int N, int D) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < B * D) {
        int b = index / D, d = index % D;
        output[(static_cast<long long>(b) * (N + 1)) * D + d] = cls[d] + pos[d];
    }
}

extern "C" void solve(const float* images, const float* patch_weight,
                      const float* patch_bias, const float* cls_token, const float* pos_embed,
                      float* output, int B, int C, int H, int W, int P, int D) {
    int N = (H / P) * (W / P), threads = 256;
    long long total = static_cast<long long>(B) * N * D;
    patch_kernel<<<static_cast<int>((total + threads - 1) / threads), threads>>>(
        images, patch_weight, patch_bias, pos_embed, output, B, C, H, W, P, D, N, W / P);
    cls_kernel<<<(B * D + threads - 1) / threads, threads>>>(cls_token, pos_embed, output, B, N, D);
    cudaDeviceSynchronize();
}
```
