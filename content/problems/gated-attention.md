---
id: gated-attention
leetgpuId: 147
titleEn: Gated Attention
titleZh: 门控注意力
description: 使用 CUDA 计算多头因果注意力，并在注意力输出之后应用 XW_g 的逐通道 Sigmoid 门控
problemUrl: https://leetgpu.com/challenges/gated-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q/K/V/X/output:[M,d_model]`，`W_g:[d_model,d_model]`。先计算行主序 `gate=sigmoid(X×W_g)`。令 `d_k=d_model/h`，每个 head 独立执行因果注意力：

```text
head=softmax(Q_h K_hᵀ/√d_k + causal_mask) V_h
output[:,head_slice]=head⊙gate[:,head_slice]
```

门控应用在注意力输出之后，不是作用于 V；`W_g` 不转置，列块对应各 head。

### 实现要求与约束

- 所有张量均为 FP32
- `1≤M≤4096`
- `d_model≤512` 且能被 h 整除，`h≤8`
- Q、K、V、X 来自有正有负的范围；W_g 经过缩放，避免 sigmoid 饱和
- 性能测试 `M=4096,d_model=512,h=8`
- 绝对误差和相对误差均为 `1e-5`

官方 `M=3,d_model=4,h=2` 示例输出为 `[[0.9525741,1.7615942,1.5,2.9242343],[2.6895983,4.1212902,3.3314223,3.4961277],[2.8680652,3.1115429,1.6336840,3.5372488]]`。

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K,const float* V,const float* X,const float* W_g,float* output,int M,int d_model,int h) {}
```

## 实现思路

gate 投影使用 16×16 shared tiled GEMM，并原位应用 sigmoid。注意力采用一个 128 线程 Block 对应 `(query,head)`；对每个 `key≤query`，lane 协作计算一次 score，再在线更新 m、l 和 head_dim 维 acc。输出归一化后乘对应 gate 通道，没有 M×M score 缓冲，也不按输出维重复 score。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
#define T 16
__global__ void gate_gemm(const float*X,const float*W,float*G,int M,int D){__shared__ float a[T][T],b[T][T];int tx=threadIdx.x,ty=threadIdx.y,r=blockIdx.y*T+ty,c=blockIdx.x*T+tx;double sum=0;for(int base=0;base<D;base+=T){int k1=base+tx,k2=base+ty;a[ty][tx]=(r<M&&k1<D)?X[r*D+k1]:0;b[ty][tx]=(k2<D&&c<D)?W[k2*D+c]:0;__syncthreads();for(int k=0;k<T;++k)sum+=double(a[ty][k])*b[k][tx];__syncthreads();}if(r<M&&c<D)G[r*D+c]=1.0f/(1.0f+expf(-float(sum)));}
__global__ void attention(const float*Q,const float*K,const float*V,const float*G,float*out,int D,int H){extern __shared__ double sh[];double*red=sh;double*acc=sh+128;int x=blockIdx.x,lane=threadIdx.x,row=x/H,h=x%H,dk=D/H,base=h*dk;double m=-CUDART_INF,l=0.0,scale=1.0/sqrt(double(dk));for(int d=lane;d<dk;d+=blockDim.x)acc[d]=0.0;__syncthreads();for(int j=0;j<=row;++j){double part=0.0;for(int d=lane;d<dk;d+=blockDim.x)part+=double(Q[size_t(row)*D+base+d])*K[size_t(j)*D+base+d];red[lane]=part;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)red[lane]+=red[lane+r];__syncthreads();}double score=red[0]*scale,newM=fmax(m,score),alpha=exp(m-newM),beta=exp(score-newM);for(int d=lane;d<dk;d+=blockDim.x)acc[d]=acc[d]*alpha+beta*V[size_t(j)*D+base+d];l=l*alpha+beta;__syncthreads();m=newM;}for(int d=lane;d<dk;d+=blockDim.x)out[size_t(row)*D+base+d]=float(acc[d]/l)*G[size_t(row)*D+base+d];}
extern "C" void solve(const float*Q,const float*K,const float*V,const float*X,const float*W_g,float*output,int M,int d_model,int h){float*G;cudaMalloc(&G,size_t(M)*d_model*sizeof(float));dim3 threads(T,T),grid((d_model+T-1)/T,(M+T-1)/T);gate_gemm<<<grid,threads>>>(X,W_g,G,M,d_model);size_t shared=size_t(128+d_model/h)*sizeof(double);attention<<<M*h,128,shared>>>(Q,K,V,G,output,d_model,h);cudaDeviceSynchronize();cudaFree(G);}
```
