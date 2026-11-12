---
id: multi-scale-deformable-attention
leetgpuId: 123
titleEn: Multi-Scale Deformable Attention
titleZh: 多尺度可变形注意力
description: 使用 CUDA 按多尺度归一化坐标执行零填充双线性采样与加权汇总
problemUrl: https://leetgpu.com/challenges/multi-scale-deformable-attention
updatedAt: 2026-11-12
---
## 题目定义

多层特征图沿空间维拼接为 `value(total_spatial,num_heads,head_dim)`。`spatial_shapes(l,2)` 给出每层 `(H,W)`。对每个 `(query,head,level,point)`，归一化坐标 `(x_norm,y_norm)` 转换为：

```text
x=x_norm*W-0.5
y=y_norm*H-0.5
```

在四个角上双线性插值；单个角越界时该角贡献零。采样结果乘题目给定的 `attn_weight` 后对 level、point 求和，权重不再归一化。

### 示例

两层为 `2×2:[[1,2],[3,4]]` 和 `1×1:[[10]]`。中心采样得到 2.5 与 10，权重各 0.5，Query 0 输出 6.25；边缘 `(1,1)` 采样只剩一个有效角，Query 1 输出 3.25。

约束为 queries≤25000、heads≤16、head_dim≤64、levels≤8、points≤8、每层 `1≤H_l,W_l≤128`，总空间≤25000；`sampling_loc∈[-0.25,1.25]`，`attn_weight∈[-5,5]`。value、sampling_loc、attn_weight、output 为 FP32，spatial_shapes 为 INT32。性能用例 queries=21760、heads=8、dim=32、4层、4点，容差 `1e-4`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* value, const int* spatial_shapes,
 const float* sampling_loc, const float* attn_weight, float* output,
 int num_queries, int num_heads, int head_dim, int num_levels, int num_points)
```

sampling_loc 最后一维顺序为 x、y；value 每个空间位置内按 head、dim 连续。层起点是此前各层 `H*W` 的前缀和。

## 实现思路

先在 host 读取最多 8 个 shape，计算 level start，再复制到小型设备数组。每个线程负责一个 `(query,head,dim)` 输出，循环 levels×points 与四个角。几何和注意力权重会在 dim 线程间重复计算，但最大额外维只有 64；整个输出有数百万线程，避免按 Query 串行。

使用 `floorf` 而不是整数截断，保证负坐标正确。每个角独立判断范围，坐标落在边界外半个像素内时仍保留有效角的部分权重。

## 参考实现

```cpp
#include <cmath>
#include <cuda_runtime.h>

__global__ void deformable_kernel(const float* value, const int* shapes, const int* starts,
                                  const float* loc, const float* weights, float* output,
                                  int Q, int H, int D, int L, int P) {
    long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(Q) * H * D;
    if (index >= total) return;
    int d = index % D;
    int h = (index / D) % H;
    int q = index / (static_cast<long long>(D) * H);
    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        int height = shapes[2 * l], width = shapes[2 * l + 1];
        for (int p = 0; p < P; ++p) {
            long long item = ((static_cast<long long>(q) * H + h) * L + l) * P + p;
            float x = loc[item * 2] * width - 0.5f;
            float y = loc[item * 2 + 1] * height - 0.5f;
            int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
            float dx = x - x0, dy = y - y0;
            float aw = weights[item];
            for (int oy = 0; oy < 2; ++oy) for (int ox = 0; ox < 2; ++ox) {
                int xi = x0 + ox, yi = y0 + oy;
                if (xi < 0 || xi >= width || yi < 0 || yi >= height) continue;
                float bilinear = (ox ? dx : 1.0f - dx) * (oy ? dy : 1.0f - dy);
                int spatial = starts[l] + yi * width + xi;
                sum += aw * bilinear * value[(static_cast<long long>(spatial) * H + h) * D + d];
            }
        }
    }
    output[index] = sum;
}

extern "C" void solve(const float* value, const int* spatial_shapes,
                      const float* sampling_loc, const float* attn_weight, float* output,
                      int num_queries, int num_heads, int head_dim, int num_levels, int num_points) {
    int host_shapes[16], host_starts[8], running = 0;
    cudaMemcpy(host_shapes, spatial_shapes, 2 * num_levels * sizeof(int), cudaMemcpyDeviceToHost);
    for (int l = 0; l < num_levels; ++l) {
        host_starts[l] = running;
        running += host_shapes[2 * l] * host_shapes[2 * l + 1];
    }
    int *shapes = nullptr, *starts = nullptr;
    cudaMalloc(&shapes, 2 * num_levels * sizeof(int)); cudaMalloc(&starts, num_levels * sizeof(int));
    cudaMemcpy(shapes, host_shapes, 2 * num_levels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(starts, host_starts, num_levels * sizeof(int), cudaMemcpyHostToDevice);
    long long total = static_cast<long long>(num_queries) * num_heads * head_dim;
    deformable_kernel<<<static_cast<int>((total + 255) / 256), 256>>>(
        value, shapes, starts, sampling_loc, attn_weight, output,
        num_queries, num_heads, head_dim, num_levels, num_points);
    cudaDeviceSynchronize(); cudaFree(starts); cudaFree(shapes);
}
```
