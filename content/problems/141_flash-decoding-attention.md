---
id: flash-decoding-attention
leetgpuId: 141
titleEn: Flash Decoding Attention
titleZh: Flash 解码注意力
description: 使用在线 Softmax 在长 KV 上计算无掩码多头解码注意力
problemUrl: https://leetgpu.com/challenges/flash-decoding-attention
updatedAt: 2026-11-12
---
## 题目定义

布局为 `Q:[T,H,D]`、`K/V:[S,H,D]`、输出 `[T,H,D]`：

```text
score[t,h,s]=dot(Q[t,h],K[s,h])/√D
output=softmax(score)×V
```

本题没有因果 mask：K/V 已表示当前可见缓存，每个 query 访问全部 S 个键。

### 实现要求与约束

- `1≤T≤16`，`1≤S≤131072`
- `1≤H≤64`，`1≤D≤256`
- 所有张量均为 FP32，输入值位于 `[-6,6]`
- 保持签名并使用稳定 Softmax
- 功能测试含零 query、大 score 和非 2 的幂
- 性能测试 `T=1,S=65536,H=32,D=128`
- 绝对误差和相对误差均为 `1e-4`

官方示例 `T=2,S=3,H=1,D=2`，两行输出约为 `[3,4]` 和 `[3.4067,4.4067]`。

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K,const float* V,float* output,int T,int S,int H,int D) {}
```

## 实现思路

一个 128 线程 Block 负责 `(t,h)`。对每个 key，所有 lane 协作计算一次 Q·K；随后按在线 Softmax 递推更新最大值 m、分母 l 和 D 维分子 acc。每个 score 只计算一次，不分配长度 S 的 score 缓冲。Shared Memory 为 `128+D` 个 double，D=256 时只需 3 KiB，适用于长 KV cache。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void decode(const float*Q,const float*K,const float*V,float*out,int S,int H,int D){extern __shared__ double sh[];double*red=sh;double*acc=sh+128;int x=blockIdx.x,lane=threadIdx.x,q=x/H,h=x%H;double m=-CUDART_INF,l=0.0,scale=1.0/sqrt(double(D));for(int d=lane;d<D;d+=blockDim.x)acc[d]=0.0;__syncthreads();for(int s=0;s<S;++s){double part=0.0;for(int d=lane;d<D;d+=blockDim.x)part+=double(Q[(size_t(q)*H+h)*D+d])*K[(size_t(s)*H+h)*D+d];red[lane]=part;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)red[lane]+=red[lane+r];__syncthreads();}double score=red[0]*scale,newM=fmax(m,score),alpha=exp(m-newM),beta=exp(score-newM);for(int d=lane;d<D;d+=blockDim.x)acc[d]=acc[d]*alpha+beta*V[(size_t(s)*H+h)*D+d];l=l*alpha+beta;__syncthreads();m=newM;}for(int d=lane;d<D;d+=blockDim.x)out[(size_t(q)*H+h)*D+d]=float(acc[d]/l);}
extern "C" void solve(const float* Q,const float* K,const float* V,float* output,int T,int S,int H,int D){size_t shared=size_t(128+D)*sizeof(double);decode<<<T*H,128,shared>>>(Q,K,V,output,S,H,D);cudaDeviceSynchronize();}
```
