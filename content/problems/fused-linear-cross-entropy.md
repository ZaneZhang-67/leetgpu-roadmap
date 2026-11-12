---
id: fused-linear-cross-entropy
leetgpuId: 129
titleEn: Fused Linear Cross Entropy
titleZh: 融合线性投影与交叉熵
description: 不物化 Logits，以词表分块和在线 LogSumExp 计算平均交叉熵
problemUrl: https://leetgpu.com/challenges/fused-linear-cross-entropy
updatedAt: 2026-11-12
---
## 题目定义

`X(N,D)` 与 `W(V,D)` 的每一词表行点积得到 logit。每行损失是稳定 LogSumExp 减目标 logit，最终输出 N 行平均：

```text
z[i,v]=Σd X[i,d]W[v,d]
loss=(1/N)Σi(logΣv exp(z[i,v])-z[i,target[i]])
```

W 是 `(V,D)`，等价于 `X Wᵀ`。targets 为 INT32。

### 示例

官方示例 logits 为 `[[1,0,1],[0,2,2]]`，两行损失约 0.8620、0.7586，平均 `loss[0]=0.8103`。

约束 `1≤N≤8192,1≤D≤4096,2≤V≤65536`，每个 `target[i]` 都满足 `0≤target[i]<V`，logit 可达数百。性能用例 `N=4096,D=1024,V=32768`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* X, const float* W, const int* targets,
                      float* loss, int N, int D, int V)
```

loss 是单元素输出，必须计算均值而不是总和。

## 实现思路

每个 Block 负责一行，线程分别扫描一部分词表。每个 logit 的 D 维点积由当前线程计算，局部维护 `(max,sumExp,targetLogit)`；随后 Block 归约最大值，再以 `exp(localMax-globalMax)` 修正局部指数和。这种流式方法只保存 N 个行损失，不分配 `N×V` logits，并能处理大 logit。

词表并行有 256 路，但点积仍是线程内 D 循环；高端实现可把词表×D 改为二维 tiled GEMM 后在线归约。此实现覆盖最大规模且无全局 logits 内存瓶颈。

## 参考实现

```cpp
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
__global__ void rows(const float*X,const float*W,const int*t,float*out,int N,int D,int V){
 __shared__ float sm[256],ss[256];int r=blockIdx.x,id=threadIdx.x;if(r>=N)return;float mx=-FLT_MAX,sum=0,target=0;
 for(int v=id;v<V;v+=256){float z=0;for(int d=0;d<D;++d)z+=X[(long long)r*D+d]*W[(long long)v*D+d];if(v==t[r])target=z;float nm=fmaxf(mx,z);sum=sum*expf(mx-nm)+expf(z-nm);mx=nm;}
 sm[id]=mx;__syncthreads();for(int o=128;o;o>>=1){if(id<o)sm[id]=fmaxf(sm[id],sm[id+o]);__syncthreads();}float gm=sm[0];__syncthreads();ss[id]=sum*expf(mx-gm);sm[id]=target;__syncthreads();for(int o=128;o;o>>=1){if(id<o){ss[id]+=ss[id+o];sm[id]+=sm[id+o];}__syncthreads();}if(id==0)out[r]=gm+logf(ss[0])-sm[0];}
__global__ void reduce(const float*x,float*y,int N){__shared__ float s[256];int t=threadIdx.x;float v=0;for(int i=t;i<N;i+=256)v+=x[i];s[t]=v;__syncthreads();for(int o=128;o;o>>=1){if(t<o)s[t]+=s[t+o];__syncthreads();}if(t==0)y[0]=s[0]/N;}
extern "C" void solve(const float*X,const float*W,const int*targets,float*loss,int N,int D,int V){float*rl;cudaMalloc(&rl,N*4);rows<<<N,256>>>(X,W,targets,rl,N,D,V);reduce<<<1,256>>>(rl,loss,N);cudaDeviceSynchronize();cudaFree(rl);}
```
