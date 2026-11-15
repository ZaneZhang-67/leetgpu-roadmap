---
id: k-means-clustering
leetgpuId: 20
titleEn: K-Means Clustering
titleZh: K 均值聚类
description: 使用 CUDA 对二维点执行固定轮数的 K-Means，完成最近质心分配、质心更新和最终标签输出
problemUrl: https://leetgpu.com/challenges/k-means-clustering
updatedAt: 2026-11-12
---
## 题目定义

给定 `sample_size` 个二维点、`k` 个初始质心和迭代次数 `max_iterations`，执行 K-Means 聚类每轮包含两个步骤：

1. 将每个点分配给欧氏距离最近的质心
2. 用簇内所有点的坐标平均值更新质心

完成指定轮数后，将最终标签写入 `labels`，最终质心分别写入 `final_centroid_x` 和 `final_centroid_y`

点坐标采用 SoA 布局：所有 x 坐标存放在 `data_x`，所有 y 坐标存放在 `data_y`质心也分别保存在 x、y 数组中除 `labels` 为 `int32` 外，其余数组都是 `float32`

### 实现要求

- 不允许使用外部库
- `solve` 函数签名保持不变
- 最终结果写入 `labels`、`final_centroid_x` 和 `final_centroid_y`

### 示例

```text
sample_size = 4，k = 2，max_iterations = 10
data_x = [1.0, 2.0, 8.0, 9.0]
data_y = [1.0, 2.0, 8.0, 9.0]
initial_centroid_x = [1.0, 8.0]
initial_centroid_y = [1.0, 8.0]
```

题面未直接列出该示例的数值输出，结果以平台参考实现为准

### 约束

- `1 ≤ sample_size ≤ 1,000,000`
- `1 ≤ k ≤ 1,000`
- 除 `labels` 为 `int32` 外，所有数组均为 `float32`
- 性能测试使用 `sample_size = 10,000`、`k = 5`、`max_iterations = 30`

## 模板代码解析

CUDA starter 为：

```cpp
#include <cuda_runtime.h>

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {}
```

七个数组均为设备指针初始质心参数没有 `const` 修饰，但参考实现只读取它们；当前质心保存在单独的临时设备数组中最后再复制到两个输出质心数组

## 实现思路

### 1. 最近质心分配

一个线程处理一个点，对全部 `k` 个质心计算平方欧氏距离：

```text
distance² = (x - centroid_x[c])² + (y - centroid_y[c])²
```

平方根不会改变距离大小关系，因此无需调用 `sqrtf`当距离严格更小时更新最佳簇；距离相同时保留编号更小的质心

### 2. 用原子操作累计簇统计量

分配标签后，需要为每个簇统计：

```text
sum_x[c] = 簇 c 中所有点的 x 坐标之和
sum_y[c] = 簇 c 中所有点的 y 坐标之和
count[c] = 簇 c 中的点数
```

多个线程可能同时更新同一簇，因此使用 `atomicAdd`

### 3. 更新质心与空簇

若 `count[c] > 0`：

```text
centroid_x[c] = sum_x[c] / count[c]
centroid_y[c] = sum_y[c] / count[c]
```

若某个簇本轮没有样本，参考实现保留它上一轮的质心，避免除零

### 4. 固定轮数的输出语义

平台参考流程每轮先按当前质心写入 `labels`，再用这些标签更新质心循环完成后直接返回，因此最终 `labels` 保留最后一轮分配结果，不再依据更新后的质心重算

### 5. 临时存储

`solve` 使用 `cudaMalloc` 创建当前质心、坐标和、计数数组每轮先用 `cudaMemset` 清空累计数组，再依次完成分配与更新所有临时分配在返回前释放

## 参考实现

```cpp
#include <cfloat>
#include <cuda_runtime.h>

__global__ void assign_and_accumulate(const float* data_x, const float* data_y, int* labels,
                                      const float* centroid_x, const float* centroid_y,
                                      float* sum_x, float* sum_y, int* counts, int sample_size,
                                      int k) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < sample_size) {
        float x = data_x[id];
        float y = data_y[id];
        float best_distance = FLT_MAX;
        int best_cluster = 0;

        for (int c = 0; c < k; ++c) {
            float dx = x - centroid_x[c];
            float dy = y - centroid_y[c];
            float distance = dx * dx + dy * dy;
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = c;
            }
        }

        labels[id] = best_cluster;
        atomicAdd(&sum_x[best_cluster], x);
        atomicAdd(&sum_y[best_cluster], y);
        atomicAdd(&counts[best_cluster], 1);
    }
}

__global__ void update_centroids(float* centroid_x, float* centroid_y, const float* sum_x,
                                 const float* sum_y, const int* counts, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < k && counts[c] > 0) {
        centroid_x[c] = sum_x[c] / counts[c];
        centroid_y[c] = sum_y[c] / counts[c];
    }
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
    float *centroid_x, *centroid_y, *sum_x, *sum_y;
    int* counts;
    cudaMalloc(&centroid_x, static_cast<size_t>(k) * sizeof(float));
    cudaMalloc(&centroid_y, static_cast<size_t>(k) * sizeof(float));
    cudaMalloc(&sum_x, static_cast<size_t>(k) * sizeof(float));
    cudaMalloc(&sum_y, static_cast<size_t>(k) * sizeof(float));
    cudaMalloc(&counts, static_cast<size_t>(k) * sizeof(int));

    cudaMemcpy(centroid_x, initial_centroid_x, static_cast<size_t>(k) * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(centroid_y, initial_centroid_y, static_cast<size_t>(k) * sizeof(float),
               cudaMemcpyDeviceToDevice);

    int threads = 256;
    int point_blocks = (sample_size + threads - 1) / threads;
    int centroid_blocks = (k + threads - 1) / threads;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        cudaMemset(sum_x, 0, static_cast<size_t>(k) * sizeof(float));
        cudaMemset(sum_y, 0, static_cast<size_t>(k) * sizeof(float));
        cudaMemset(counts, 0, static_cast<size_t>(k) * sizeof(int));

        assign_and_accumulate<<<point_blocks, threads>>>(
            data_x, data_y, labels, centroid_x, centroid_y, sum_x, sum_y, counts, sample_size, k);
        update_centroids<<<centroid_blocks, threads>>>(centroid_x, centroid_y, sum_x, sum_y,
                                                       counts, k);
    }

    cudaMemcpy(final_centroid_x, centroid_x, static_cast<size_t>(k) * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, centroid_y, static_cast<size_t>(k) * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cudaFree(centroid_x);
    cudaFree(centroid_y);
    cudaFree(sum_x);
    cudaFree(sum_y);
    cudaFree(counts);
}
```
