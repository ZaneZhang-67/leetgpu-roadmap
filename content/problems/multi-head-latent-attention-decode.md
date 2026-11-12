---
id: multi-head-latent-attention-decode
leetgpuId: 114
titleEn: Multi-Head Latent Attention Decode
titleZh: 多头潜变量注意力解码
description: 使用 CUDA 通过权重吸收、潜变量缓存和解耦 RoPE 键完成 MLA 解码注意力
problemUrl: https://leetgpu.com/challenges/multi-head-latent-attention-decode
updatedAt: 2026-11-12
---
## 题目定义

布局：`q:[H,head_dim+rope_dim]`，`kv_cache:[S,rank+rope_dim]`，`W_UK:[H,head_dim,rank]`，`W_UV:[H,rank,head_dim]`，输出 `[H,head_dim]`。

```text
q_latent = q_nope × W_UK
score[h,s] = (q_latent·c_kv[s] + q_pe·k_pe[s]) / √(head_dim+rope_dim)
attn = softmax(score)
latent_out = attn × c_kv
output = latent_out × W_UV
```

### 实现要求

- 保持 starter 签名
- 对完整 `seq_len` 做稳定 Softmax
- 不显式展开每个位置的完整 K/V

### 示例与约束

官方示例为 `H=2,S=2,rank=2,head_dim=2,rope_dim=1`。

- `1≤num_heads≤128`
- `1≤seq_len≤8192`
- `1≤kv_lora_rank≤512`
- `1≤head_dim≤128`
- `1≤rope_dim≤64`
- 所有张量均为 FP32
- 功能测试覆盖零 query、零 cache 与非 2 的幂尺寸
- 性能测试为 `H=128,S=4096,rank=512,head_dim=128,rope_dim=64`
- 绝对误差和相对误差均为 `1e-3`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// q, kv_cache, W_UK, W_UV, output are device pointers
extern "C" void solve(
    const float* q, const float* kv_cache,
    const float* W_UK, const float* W_UV,
    float* output, int num_heads, int seq_len,
    int kv_lora_rank, int head_dim, int rope_dim
) {}
```

## 实现思路

分五个并行 kernel：一维 Grid 展平元素，计算每个 head 的 `q_latent`；二维 Grid 计算所有 head-position score；每个 head 一个 Block 完成稳定 Softmax；二维 Grid 用概率直接计算每个 `(head,latent)` 的加权缓存，再用一维 Grid 展平 `(head,output_dim)` 上投影到输出。这样性能规模有数十万线程参与，而不是让单线程串行完成整个 head。

中间缓冲区为 `H×rank` 的 q/latent 和 `H×S` 的 score/probability。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void absorb_query(const float* q,const float* W,float* ql,int H,int hd,int rd,int rank){
 int i=blockIdx.x*blockDim.x+threadIdx.x,total=H*rank;if(i>=total)return;int h=i/rank,r=i%rank;double s=0;for(int k=0;k<hd;++k)s+=double(q[h*(hd+rd)+k])*W[(h*hd+k)*rank+r];ql[i]=float(s);
}
__global__ void make_scores(const float* q,const float* cache,const float* ql,float* scores,int H,int S,int rank,int hd,int rd){
 int i=blockIdx.x*blockDim.x+threadIdx.x,total=H*S;if(i>=total)return;int h=i/S,p=i%S;double s=0;for(int r=0;r<rank;++r)s+=double(ql[h*rank+r])*cache[p*(rank+rd)+r];for(int r=0;r<rd;++r)s+=double(q[h*(hd+rd)+hd+r])*cache[p*(rank+rd)+rank+r];scores[i]=float(s/sqrt(double(hd+rd)));
}
__global__ void softmax_rows(float* scores,int S){
 extern __shared__ float sh[];int h=blockIdx.x,t=threadIdx.x;float m=-CUDART_INF_F;for(int i=t;i<S;i+=blockDim.x)m=fmaxf(m,scores[h*S+i]);sh[t]=m;__syncthreads();for(int s=blockDim.x/2;s;s>>=1){if(t<s)sh[t]=fmaxf(sh[t],sh[t+s]);__syncthreads();}m=sh[0];float sum=0;for(int i=t;i<S;i+=blockDim.x){float v=expf(scores[h*S+i]-m);scores[h*S+i]=v;sum+=v;}sh[t]=sum;__syncthreads();for(int s=blockDim.x/2;s;s>>=1){if(t<s)sh[t]+=sh[t+s];__syncthreads();}for(int i=t;i<S;i+=blockDim.x)scores[h*S+i]/=sh[0];
}
__global__ void latent_attention(const float* p,const float* cache,float* lo,int H,int S,int rank,int rd){int i=blockIdx.x*blockDim.x+threadIdx.x,total=H*rank;if(i>=total)return;int h=i/rank,r=i%rank;double s=0;for(int j=0;j<S;++j)s+=double(p[h*S+j])*cache[j*(rank+rd)+r];lo[i]=float(s);}
__global__ void up_project(const float* lo,const float* W,float* out,int H,int rank,int hd){int i=blockIdx.x*blockDim.x+threadIdx.x,total=H*hd;if(i>=total)return;int h=i/hd,k=i%hd;double s=0;for(int r=0;r<rank;++r)s+=double(lo[h*rank+r])*W[(h*rank+r)*hd+k];out[i]=float(s);}

extern "C" void solve(const float* q,const float* kv_cache,const float* W_UK,const float* W_UV,float* output,int num_heads,int seq_len,int kv_lora_rank,int head_dim,int rope_dim){
 int t=256;float *ql,*scores,*lo;cudaMalloc(&ql,size_t(num_heads)*kv_lora_rank*4);cudaMalloc(&scores,size_t(num_heads)*seq_len*4);cudaMalloc(&lo,size_t(num_heads)*kv_lora_rank*4);
 absorb_query<<<(num_heads*kv_lora_rank+t-1)/t,t>>>(q,W_UK,ql,num_heads,head_dim,rope_dim,kv_lora_rank);
 make_scores<<<(num_heads*seq_len+t-1)/t,t>>>(q,kv_cache,ql,scores,num_heads,seq_len,kv_lora_rank,head_dim,rope_dim);
 softmax_rows<<<num_heads,t,t*4>>>(scores,seq_len);
 latent_attention<<<(num_heads*kv_lora_rank+t-1)/t,t>>>(scores,kv_cache,lo,num_heads,seq_len,kv_lora_rank,rope_dim);
 up_project<<<(num_heads*head_dim+t-1)/t,t>>>(lo,W_UV,output,num_heads,kv_lora_rank,head_dim);
 cudaDeviceSynchronize();cudaFree(lo);cudaFree(scores);cudaFree(ql);
}
```
