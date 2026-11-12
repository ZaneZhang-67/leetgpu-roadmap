---
id: block-sparse-kv-selection-attention
leetgpuId: 119
titleEn: Block-Sparse KV Selection Attention
titleZh: 块稀疏 KV 选择注意力
description: 按均值 Key 对连续缓存块评分并确定性选择 Top-K，再对选中位置执行注意力
problemUrl: https://leetgpu.com/challenges/block-sparse-kv-selection-attention
updatedAt: 2026-11-12
---
## 题目定义

每个 Head 有一个 decode Query，K/V 为 `[heads,seq_len,head_dim]`。把序列按连续的 `block_size` 分块，尾块只平均真实位置；块评分是 `Q·mean(K_block)`，不缩放。每个 Head 选择最高的 `min(num_selected,num_blocks)` 块，分数相同时块索引小者优先。

选择后，将各选中块中的全部真实位置合并，使用 `1/sqrt(head_dim)` 的缩放分数做一个统一 Softmax，再加权 V。这里没有额外 causal mask：输入 K/V 已是当前 decode step 可用的缓存。

### 官方示例

```text
heads=2, seq_len=4, dim=2, block_size=2, selected=1
head0 Q=[1,0] selects block0
head1 Q=[0,1] selects block1
output=[[2,3],[6,7]]
```

### 完整约束

- `1 ≤ num_heads ≤ 64`，`1 ≤ seq_len ≤ 8192`
- `1 ≤ head_dim ≤ 128`，`1 ≤ block_size ≤ 256`
- `num_selected ≥ 1`；超过块数时选择全部块
- 所有张量为 FP32
- 性能测试为 `heads=32,seq_len=8192,dim=128,block_size=64,selected=32`
- 判定容差为 `1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int num_heads, int seq_len,
                      int head_dim, int block_size, int num_selected) {}
```

块索引由逻辑位置直接计算：块 `b` 覆盖 `[b*block_size,min((b+1)*block_size,seq_len))`。平台不传选择索引，需要实现评分和确定性 Top-K。

## 实现思路

一个 Block 处理一个 Head。线程并行计算各块均值分数，尾块上界使用 `min(end,seq_len)`。块数最大为 8192（`block_size=1` 是正式边界），因此把 `(score,index)` 存入全局临时数组，并对每个 Head 做确定性的并行 bitonic sort；比较器按 score 降序、块索引升序。排序长度补到下一个 2 的幂，padding 使用负无穷。

排序后前 `min(num_selected,num_blocks)` 个块写入选择标记。接着在选中位置上计算统一的稳定 Softmax，权重保存在 `[heads,seq_len]` 临时数组；最后线程按 feature 并行累加 V。选择由 `O(NB log²NB)` 比较完成，不再做 `NB²` rank；Block 内 Shared Memory 只用于归约，最大规模合法。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
struct Pair{float score;int index;};
__global__ void score_blocks(const float*Q,const float*K,Pair*p,int S,int D,int BS,int NB,int padded){int h=blockIdx.y,b=blockIdx.x*blockDim.x+threadIdx.x;if(b<padded){Pair v;v.index=b;if(b<NB){int begin=b*BS,end=min(begin+BS,S),count=end-begin;float z=0;for(int j=0;j<D;++j){float mean=0;for(int t=begin;t<end;++t)mean+=K[(h*S+t)*D+j];z+=Q[h*D+j]*(mean/count);}v.score=z;}else{v.score=-CUDART_INF_F;v.index=0x7fffffff;}p[h*padded+b]=v;}}
__device__ bool worse(Pair a,Pair b){return a.score<b.score||(a.score==b.score&&a.index>b.index);}
__global__ void bitonic(Pair*p,int padded,int j,int k){int h=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=padded)return;int ix=i^j;if(ix>i){Pair a=p[h*padded+i],b=p[h*padded+ix];bool up=(i&k)==0;bool sw=up?worse(a,b):worse(b,a);if(sw){p[h*padded+i]=b;p[h*padded+ix]=a;}}}
__global__ void mark(const Pair*p,unsigned char*chosen,int NB,int padded,int keep){int h=(int)blockIdx.x,tid=(int)threadIdx.x;for(int i=tid;i<NB;i+=(int)blockDim.x)chosen[h*NB+i]=0;__syncthreads();for(int r=tid;r<keep;r+=(int)blockDim.x)chosen[h*NB+p[h*padded+r].index]=1;}
__global__ void attend(const float*Q,const float*K,const float*V,const unsigned char*chosen,float*w,float*out,int S,int D,int BS){int h=(int)blockIdx.x,tid=(int)threadIdx.x,threads=(int)blockDim.x,NB=(S+BS-1)/BS;__shared__ float red[128],den;float scale=rsqrtf((float)D),local=-CUDART_INF_F;for(int t=tid;t<S;t+=threads)if(chosen[h*NB+t/BS]){float z=0;for(int j=0;j<D;++j)z+=Q[h*D+j]*K[(h*S+t)*D+j];w[h*S+t]=z*scale;local=fmaxf(local,w[h*S+t]);}red[tid]=local;__syncthreads();for(int off=64;off;off>>=1){if(tid<off)red[tid]=fmaxf(red[tid],red[tid+off]);__syncthreads();}float maximum=red[0];__syncthreads();float sum=0;for(int t=tid;t<S;t+=threads)if(chosen[h*NB+t/BS]){w[h*S+t]=expf(w[h*S+t]-maximum);sum+=w[h*S+t];}red[tid]=sum;__syncthreads();for(int off=64;off;off>>=1){if(tid<off)red[tid]+=red[tid+off];__syncthreads();}if(tid==0)den=red[0];__syncthreads();for(int j=tid;j<D;j+=threads){float z=0;for(int t=0;t<S;++t)if(chosen[h*NB+t/BS])z+=w[h*S+t]*V[(h*S+t)*D+j];out[h*D+j]=z/den;}}
extern "C" void solve(const float*Q,const float*K,const float*V,float*out,int H,int S,int D,int BS,int selected){int NB=(S+BS-1)/BS,padded=1;while(padded<NB)padded<<=1;Pair*p;unsigned char*chosen;float*w;cudaMalloc(&p,(size_t)H*padded*sizeof(Pair));cudaMalloc(&chosen,(size_t)H*NB);cudaMalloc(&w,(size_t)H*S*4);score_blocks<<<dim3((padded+255)/256,H),256>>>(Q,K,p,S,D,BS,NB,padded);for(int k=2;k<=padded;k<<=1)for(int j=k>>1;j;j>>=1)bitonic<<<dim3((padded+255)/256,H),256>>>(p,padded,j,k);int keep=selected<NB?selected:NB;mark<<<H,256>>>(p,chosen,NB,padded,keep);attend<<<H,128>>>(Q,K,V,chosen,w,out,S,D,BS);cudaDeviceSynchronize();cudaFree(p);cudaFree(chosen);cudaFree(w);}
```
