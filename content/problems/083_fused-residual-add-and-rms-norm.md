---
id: fused-residual-add-and-rms-norm
leetgpuId: 83
titleEn: Fused Residual Add and RMS Norm
titleZh: 融合残差加法与 RMS 归一化
description: 使用单个 CUDA kernel 融合逐元素残差加法、逐行 RMS 归一化和特征缩放
problemUrl: https://leetgpu.com/challenges/fused-residual-add-and-rms-norm
updatedAt: 2026-11-12
---
## 题目定义

`x` 与 `residual` 的形状均为 `(N,C)`，`weight` 的形状为 `(C,)`。对每个 token 行独立计算：

```text
z[i,j]   = x[i,j] + residual[i,j]
rms[i]   = sqrt(Σ z[i,j]^2 / C + eps)
out[i,j] = z[i,j] / rms[i] × weight[j]
```

中间结果 `z` 不写入全局内存，残差加法、归约和缩放需要融合在一个 kernel 中。

### 示例

```text
N=1, C=4
x        = [1.0, 0.0,-1.0, 2.0]
residual = [0.5, 1.5, 0.5,-0.5]
weight   = [1.0, 1.0, 1.0, 1.0]
eps      = 1e-5
z        = [1.5, 1.5,-0.5, 1.5]
rms      = sqrt(1.75 + 1e-5) ≈ 1.32288
out      ≈ [1.1339,1.1339,-0.3780,1.1339]
```

**约束与测试**

- `1≤N,C≤65,536`，所有张量均为 FP32
- `x`、`residual` 位于 `[-10,10]`，`weight` 位于 `[0.5,1.5]`，`eps=1e-5`
- 测试覆盖 `C=1`、零 `x`、零 residual、全负值和非二次幂 `C=255`
- 性能测试为 `N=C=4096`，绝对与相对误差容限均为 `1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {}
```

starter 只提供空的 CPU 入口。`weight[j]` 由所有行共享，`eps` 按值传入；作答时需要自行实现融合 kernel 和启动逻辑。

## 实现思路

每个 Block 负责一行。第一遍由各线程跨步读取本行的 `x+residual`，累加平方和，再通过共享内存完成 Block 归约。归约使用 FP32，与正式运行时的数据类型一致。

线程 0 计算 `rsqrt(sum/C+eps)`，并将结果写入共享内存。同步后，各线程在第二遍重新计算 `z`，直接写出 `z*inv_rms*weight[col]`。这样既不需要全局 `z` 缓冲区，也能保证 RMS 使用整行的全部 `C` 个元素。`C` 无须是二次幂，越界线程以零参与归约。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_rms_kernel(const float* x, const float* residual,
                                 const float* weight, float* out,
                                 int N, int C, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float shared[];
    int tid = static_cast<int>(threadIdx.x);
    int block_size = static_cast<int>(blockDim.x);
    float sum = 0.0f;
    size_t base = (size_t)row * C;
    for (int col = tid; col < C; col += block_size) {
        float z = x[base + col] + residual[base + col];
        sum += z * z;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    if (tid == 0) shared[0] = rsqrtf(shared[0] / C + eps);
    __syncthreads();
    float inv_rms = shared[0];
    for (int col = tid; col < C; col += block_size) {
        float z = x[base + col] + residual[base + col];
        out[base + col] = z * inv_rms * weight[col];
    }
}

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    int threads = 256;
    fused_rms_kernel<<<N, threads, threads * sizeof(float)>>>(
        x, residual, weight, out, N, C, eps);
    cudaDeviceSynchronize();
}
```
