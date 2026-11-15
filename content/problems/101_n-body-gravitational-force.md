---
id: n-body-gravitational-force
leetgpuId: 101
titleEn: N-body Gravitational Force
titleZh: N 体引力
description: 使用 CUDA 计算三维粒子之间带软化项的全对全引力
problemUrl: https://leetgpu.com/challenges/n-body-gravitational-force
updatedAt: 2026-11-12
---
## 题目定义

给定 `N` 个三维位置和质量，计算每个粒子受到的合力：

```text
F_i = Σ m_j (r_j-r_i) / (|r_j-r_i|²+eps²)^(3/2)
eps = 1e-3
```

运行时会遍历所有 `j`，包括 `j=i`。自作用与重合粒子的位移都为零，因此两者的贡献也为零。位置和输出均按 `(N,3)` 行主序存储

### 示例

```text
positions = [[0,0,0], [0,3,4]]
masses    = [2,1]
forces    = [[0,0.024,0.032], [0,-0.048,-0.064]]
```

**约束与测试**

- `1≤N≤65,536`，位置在 `[-10,10]`，质量满足 `0<mass≤100`
- positions、masses、forces 均为 FP32
- 测试覆盖 `N=1`、负坐标、重合粒子、二次幂与非二次幂规模
- 性能测试 `N=8192`，绝对与相对误差容限均为 `1e-2`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// positions, masses, forces are device pointers
extern "C" void solve(const float* positions, const float* masses, float* forces, int N) {}
```

官方 starter 中只有空入口。`positions[3*i+k]` 表示粒子 `i` 的第 `k` 个坐标，`masses[i]` 表示源粒子的质量，结果写入 `forces`

## 实现思路

每个线程处理一个目标粒子 `i`，并按 `j=0..N-1` 累加三个方向的贡献。分母 `(dist²+1e-6)^(3/2)` 可写成 `dist2*sqrt(dist2)`；使用 `rsqrtf` 后计算 `inv³`，可以减少开方和除法

由于每个线程都要读取全部源粒子，复杂度为 `O(N²)`。性能规模约 6700 万对；相邻线程会读取相同的 `j`，这些访问可由缓存复用。所有线性偏移均使用 `size_t`

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void nbody_kernel(const float* positions, const float* masses,
                             float* forces, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        size_t ib = (size_t)i * 3;
        float ix = positions[ib], iy = positions[ib + 1], iz = positions[ib + 2];
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int j = 0; j < N; ++j) {
            size_t jb = (size_t)j * 3;
            float dx = positions[jb] - ix;
            float dy = positions[jb + 1] - iy;
            float dz = positions[jb + 2] - iz;
            float dist2 = dx * dx + dy * dy + dz * dz + 1.0e-6f;
            float inv = rsqrtf(dist2);
            float factor = masses[j] * inv * inv * inv;
            fx += factor * dx;
            fy += factor * dy;
            fz += factor * dz;
        }
        forces[ib] = fx; forces[ib + 1] = fy; forces[ib + 2] = fz;
    }
}

// positions, masses, forces are device pointers
extern "C" void solve(const float* positions, const float* masses, float* forces, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    nbody_kernel<<<blocks, threads>>>(positions, masses, forces, N);
    cudaDeviceSynchronize();
}
```
