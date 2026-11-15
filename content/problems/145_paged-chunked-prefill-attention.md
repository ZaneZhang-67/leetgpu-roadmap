---
id: paged-chunked-prefill-attention
leetgpuId: 145
titleEn: Paged Chunked Prefill Attention
titleZh: 分页分块预填充注意力
description: 使用 CUDA 按 block table 访问分页 KV Cache，并应用 chunk 绝对位置因果边界和 GQA 映射
problemUrl: https://leetgpu.com/challenges/paged-chunked-prefill-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q/output:[total_q,Hq,D]`，物理 cache 为 `[pool,block_size,Hkv,D]`。序列 s 的 query 范围是 `[cu[s],cu[s+1])`，context 长度为 `ctx`。chunk 是上下文最后 `q_len` 个 token，因此局部 query i 的绝对位置：

```text
q_pos=ctx-q_len+i
```

只允许逻辑 key `0..q_pos`。逻辑 key 经 `logical_block=k/block_size`、slot=`k%block_size` 和 `block_table[s,logical_block]` 映射物理 cache。GQA 映射为 `kv_head=q_head/(Hq/Hkv)`。

### 实现要求与约束

- `1≤num_seqs≤64`，每个 `1≤q_len≤context_len≤8192`，`total_q≤8192`
- `1≤Hkv≤Hq≤64`，且 Hq 是 Hkv 的倍数
- D 为偶数，`2≤D≤256`
- block size 是 2 的幂且位于 `[1,64]`
- block table 的尾部项和末页 context 之后的槽位都是无关数据，不能读取
- 物理页可任意散布；输入来自标准正态分布，Q 可能放大到 score 约为 ±100
- 性能 workload 为 4 个 512-token chunk 加 28 个 decode 序列，`total_q=2076,Hq=32,Hkv=8,D=128,block_size=16`
- 绝对误差和相对误差均为 `1e-3`

官方示例的两条 query 分别访问逻辑键 `0..1` 与 `0..2`；对应输出为 `[[1.5,1.5],[1.5,1.5]]` 和 `[[2,2],[1.6626,1.6626]]`。

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K_cache,const float* V_cache,const int* block_table,const int* context_lens,const int* cu_seqlens_q,float* output,int num_seqs,int total_q,int num_q_heads,int num_kv_heads,int head_dim,int block_size,int max_blocks_per_seq) {}
```

## 实现思路

先用并行 kernel 为每个全局 query 找到所属 sequence、chunk 局部索引和 causal key count。随后一个 128 线程 Block 负责 `(query,q_head)`：对每个合法逻辑 key，lane 协作计算一次 score，再在线更新 m、l 与 D 维 acc。物理索引从 `size_t(physical_block)×block_size` 开始计算，避免页号乘法溢出。整个流程不会访问 `key>=ctx` 或 `key>q_pos`，也不会按输出维重复 score。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void locate(const int*cu,const int*ctx,int*seq,int*limit,int n,int total){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=total)return;int s=0;while(s+1<n&&q>=cu[s+1])++s;int qlen=cu[s+1]-cu[s];seq[q]=s;limit[q]=ctx[s]-qlen+(q-cu[s])+1;}
__global__ void paged(const float*Q,const float*K,const float*V,const int*table,const int*seq,const int*limit,float*out,int Hq,int Hkv,int D,int bs,int maxb){extern __shared__ double sh[];double*red=sh;double*acc=sh+128;int x=blockIdx.x,lane=threadIdx.x,q=x/Hq,h=x%Hq,s=seq[q],L=limit[q],kh=h/(Hq/Hkv);double m=-CUDART_INF,l=0.0,scale=1.0/sqrt(double(D));for(int d=lane;d<D;d+=blockDim.x)acc[d]=0.0;__syncthreads();for(int k=0;k<L;++k){int pb=table[s*maxb+k/bs],slot=k%bs;size_t base=(size_t(pb)*bs+slot)*Hkv+kh;double part=0.0;for(int d=lane;d<D;d+=blockDim.x)part+=double(Q[(size_t(q)*Hq+h)*D+d])*K[base*D+d];red[lane]=part;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)red[lane]+=red[lane+r];__syncthreads();}double score=red[0]*scale,newM=fmax(m,score),alpha=exp(m-newM),beta=exp(score-newM);for(int d=lane;d<D;d+=blockDim.x)acc[d]=acc[d]*alpha+beta*V[base*D+d];l=l*alpha+beta;__syncthreads();m=newM;}for(int d=lane;d<D;d+=blockDim.x)out[(size_t(q)*Hq+h)*D+d]=float(acc[d]/l);}
extern "C" void solve(const float*Q,const float*K_cache,const float*V_cache,const int*block_table,const int*context_lens,const int*cu_seqlens_q,float*output,int num_seqs,int total_q,int num_q_heads,int num_kv_heads,int head_dim,int block_size,int max_blocks_per_seq){int *seq,*lim;cudaMalloc(&seq,size_t(total_q)*sizeof(int));cudaMalloc(&lim,size_t(total_q)*sizeof(int));locate<<<(total_q+255)/256,256>>>(cu_seqlens_q,context_lens,seq,lim,num_seqs,total_q);size_t shared=size_t(128+head_dim)*sizeof(double);paged<<<total_q*num_q_heads,128,shared>>>(Q,K_cache,V_cache,block_table,seq,lim,output,num_q_heads,num_kv_heads,head_dim,block_size,max_blocks_per_seq);cudaDeviceSynchronize();cudaFree(lim);cudaFree(seq);}
```
