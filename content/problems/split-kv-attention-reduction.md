---
id: split-kv-attention-reduction
leetgpuId: 128
titleEn: Split-KV Attention Reduction
titleZh: Split-KV 注意力归约
description: 使用各分片的 LSE 稳定权重合并 Split-KV 注意力局部输出
problemUrl: https://leetgpu.com/challenges/split-kv-attention-reduction
updatedAt: 2026-11-12
---
## 题目定义

`partial_out:[num_splits,num_heads,head_dim]` 是每个 KV 分片内部已经归一化的注意力输出，`partial_lse:[num_splits,num_heads]` 是对应 score 的 log-sum-exp。对每个 head：

```text
m=max_s lse[s,h]
w[s]=exp(lse[s,h]-m)
output[h,d]=Σ_s w[s]×partial_out[s,h,d] / Σ_s w[s]
```

不能直接平均 partial output；LSE 决定每片在完整 Softmax 中的质量。

### 实现要求与约束

保持签名，FP32 输出。正式范围为 `1≤num_splits≤256`、`1≤num_heads≤128`、`1≤head_dim≤256`；覆盖单 split、相等 LSE、非 2 的幂 split/head_dim。性能测试 `splits=128,heads=64,head_dim=128`，容差 `1e-4`。

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* partial_out,const float* partial_lse,float* output,int num_splits,int num_heads,int head_dim) {}
```

## 实现思路

一个 Block 负责一个 head。线程先并行归约最大 LSE，再归约指数权重和；随后各线程跨步计算输出维。Shared Memory 仅需 256 个 float，最大值平移保证指数稳定。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void merge(const float*po,const float*pl,float*out,int S,int H,int D){__shared__ float sh[256];int h=blockIdx.x,t=threadIdx.x;float m=-CUDART_INF_F;for(int s=t;s<S;s+=blockDim.x)m=fmaxf(m,pl[s*H+h]);sh[t]=m;__syncthreads();for(int k=128;k;k>>=1){if(t<k)sh[t]=fmaxf(sh[t],sh[t+k]);__syncthreads();}m=sh[0];float z=0;for(int s=t;s<S;s+=blockDim.x)z+=expf(pl[s*H+h]-m);sh[t]=z;__syncthreads();for(int k=128;k;k>>=1){if(t<k)sh[t]+=sh[t+k];__syncthreads();}z=sh[0];for(int d=t;d<D;d+=blockDim.x){float v=0;for(int s=0;s<S;++s)v+=expf(pl[s*H+h]-m)*po[(s*H+h)*D+d];out[h*D+d]=v/z;}}
extern "C" void solve(const float* partial_out,const float* partial_lse,float* output,int num_splits,int num_heads,int head_dim){merge<<<num_heads,256>>>(partial_out,partial_lse,output,num_splits,num_heads,head_dim);cudaDeviceSynchronize();}
```
