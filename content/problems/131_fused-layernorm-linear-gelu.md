---
id: fused-layernorm-linear-gelu
leetgpuId: 131
titleEn: Fused LayerNorm Linear GELU
titleZh: 融合 LayerNorm、Linear 与 GELU
description: 以稳定统计和共享内存分块实现 LayerNorm、线性投影及精确 GELU
problemUrl: https://leetgpu.com/challenges/fused-layernorm-linear-gelu
updatedAt: 2026-11-12
---
## 题目定义

对 `x(M,K)` 每行计算 biased variance（除以 K），再应用 gamma、beta，乘 `W(K,N)`、加 bias，并使用精确 erf GELU：

```text
xn=(x-mean)/sqrt(var+eps)*gamma+beta
h=xn W+bias
y=0.5*h*(1+erf(h/sqrt(2)))
```

不得换成 tanh 近似。W 的方向是 `(K,N)`。

### 示例

官方 `M=K=N=2` 示例中，归一化行为 `[-1,1]`、`[0,0]`，线性结果 `[[-1,4],[0,1]]`，输出约 `[[-0.1587,3.9999],[0,0.8413]]`。

约束 `M≤8192`、`K,N≤8192`、`0<eps≤1`；输入可带约 -1000 的公共偏移。性能用例三维均为 4096。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* x, const float* gamma, const float* beta,
 const float* W, const float* bias, float* y, int M, int K, int N, float eps)
```

gamma、beta 长度均为 K，bias 长度 N。

## 实现思路

先用每行一个 Block 的 Welford 合并统计得到 mean 与方差，避免 `E[x²]-E[x]²` 在大公共偏移下相消。随后 tiled GEMM 在加载 x tile 时完成标准化和 gamma/beta，W 按 `(K,N)` 载入 Shared Memory；bias 和 erf GELU 在 epilogue 融合。仅保存 `2M` 个统计量，不物化 normalized x 或 pre-activation。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <cmath>
struct Stat{float mean,m2;int n;};
__device__ Stat merge(Stat a,Stat b){if(!a.n)return b;if(!b.n)return a;float d=b.mean-a.mean;int n=a.n+b.n;return{a.mean+d*b.n/n,a.m2+b.m2+d*d*a.n*b.n/n,n};}
__global__ void stats(const float*x,float*mean,float*var,int K){__shared__ Stat s[256];int r=blockIdx.x,t=threadIdx.x;Stat a={0,0,0};for(int k=t;k<K;k+=256){float v=x[(long long)r*K+k];Stat z={v,0,1};a=merge(a,z);}s[t]=a;__syncthreads();for(int o=128;o;o>>=1){if(t<o)s[t]=merge(s[t],s[t+o]);__syncthreads();}if(t==0){mean[r]=s[0].mean;var[r]=s[0].m2/K;}}
__global__ void fused(const float*x,const float*g,const float*be,const float*W,const float*b,float*y,const float*mean,const float*var,int M,int K,int N,float eps){
 __shared__ float A[16][16],B[16][16];int tx=threadIdx.x,ty=threadIdx.y,r=blockIdx.y*16+ty,c=blockIdx.x*16+tx;float sum=0,inv=r<M?rsqrtf(var[r]+eps):0;
 for(int k0=0;k0<K;k0+=16){int ka=k0+tx,kb=k0+ty;A[ty][tx]=(r<M&&ka<K)?(x[(long long)r*K+ka]-mean[r])*inv*g[ka]+be[ka]:0;B[ty][tx]=(kb<K&&c<N)?W[(long long)kb*N+c]:0;__syncthreads();
  #pragma unroll
  for(int k=0;k<16;++k)sum+=A[ty][k]*B[k][tx];__syncthreads();}
 if(r<M&&c<N){float h=sum+b[c];y[(long long)r*N+c]=.5f*h*(1.0f+erff(h*.7071067811865475f));}}
extern "C" void solve(const float*x,const float*gamma,const float*beta,const float*W,const float*bias,float*y,int M,int K,int N,float eps){float *m,*v;cudaMalloc(&m,M*4);cudaMalloc(&v,M*4);stats<<<M,256>>>(x,m,v,K);dim3 t(16,16),bl((N+15)/16,(M+15)/16);fused<<<bl,t>>>(x,gamma,beta,W,bias,y,m,v,M,K,N,eps);cudaDeviceSynchronize();cudaFree(v);cudaFree(m);}
```
