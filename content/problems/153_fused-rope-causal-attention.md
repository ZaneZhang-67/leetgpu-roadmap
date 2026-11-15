---
id: fused-rope-causal-attention
leetgpuId: 153
titleEn: Fused RoPE Causal Attention
titleZh: 融合 RoPE 因果注意力
description: 使用 CUDA 融合 rotate-half RoPE、GQA 映射及稳定因果注意力
problemUrl: https://leetgpu.com/challenges/fused-rope-causal-attention
updatedAt: 2026-11-12
---
## 题目定义

`Q:[Hq,S,D]`、`K/V:[Hkv,S,D]`，`cos/sin:[S,D/2]`，输出 `[Hq,S,D]`。采用 Llama/GPT-NeoX 的 rotate-half 配对：维 k 与 `k+D/2` 组成一对，位置 p 使用 `cos[p,k]`、`sin[p,k]`：

```text
first'=first*cos-second*sin
second'=second*cos+first*sin
```

Q 与 K 都旋转，V 不旋转。查询头 h 使用 `kv_head=h/(Hq/Hkv)`，位置 i 只访问 `0≤j≤i`。score 缩放为 `1/√D`。

### 实现要求与约束

- 所有张量均为 FP32，输入不可修改
- `1≤Hkv≤Hq≤32`，且 Hq 能被 Hkv 整除
- `1≤S≤4096`
- D 为偶数，`2≤D≤128`
- `cos/sin` 值位于 `[-1,1]`
- 每个输出元素都必须写入
- 性能测试 `Hq=16,Hkv=4,S=4096,D=64`
- 绝对误差和相对误差均为 `1e-5`

官方示例为 `Hq=Hkv=1,S=3,D=4`，按前后半区配对旋转；若误用相邻维交错配对，结果不同。

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K,const float* V,const float* cos,const float* sin,float* output,int num_q_heads,int num_kv_heads,int seq_len,int head_dim) {}
```

## 实现思路

先用并行 kernel 生成旋转后的 Q/K，避免在每个 score 与输出维中重复 RoPE。随后一个 128 线程 Block 负责 `(query_head,position)`：对每个因果 key 协作计算一次点积并把 score 存入 Shared Memory，归约最大值，并在分母阶段将未归一化权重回写 score；输出维直接复用权重并除以总分母。最大 score 缓冲 4096 个 float，连同 128 个归约槽约 16.5 KiB；性能规模可直接启动。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void rotate(const float*in,const float*c,const float*s,float*out,int H,int S,int D){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=H*S*D;if(idx>=total)return;int d=idx%D,p=(idx/D)%S,half=D/2,k=d<half?d:d-half;size_t base=size_t(idx/D)*D;float first=in[base+k],second=in[base+k+half],co=c[p*half+k],si=s[p*half+k];out[idx]=d<half?first*co-second*si:second*co+first*si;}
__global__ void causal(const float*Q,const float*K,const float*V,float*out,int Hq,int Hkv,int S,int D){extern __shared__ float sm[];float*score=sm;float*reduce=sm+S;int x=blockIdx.x,lane=threadIdx.x,h=x/S,i=x%S,kh=h/(Hq/Hkv);float scale=rsqrtf(float(D));for(int j=0;j<=i;++j){reduce[lane]=lane<D?Q[(size_t(h)*S+i)*D+lane]*K[(size_t(kh)*S+j)*D+lane]:0.0f;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)reduce[lane]+=reduce[lane+r];__syncthreads();}if(lane==0)score[j]=reduce[0]*scale;__syncthreads();}float m=-CUDART_INF_F;for(int j=lane;j<=i;j+=blockDim.x)m=fmaxf(m,score[j]);reduce[lane]=m;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)reduce[lane]=fmaxf(reduce[lane],reduce[lane+r]);__syncthreads();}m=reduce[0];float z=0;for(int j=lane;j<=i;j+=blockDim.x){score[j]=expf(score[j]-m);z+=score[j];}reduce[lane]=z;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)reduce[lane]+=reduce[lane+r];__syncthreads();}z=reduce[0];for(int d=lane;d<D;d+=blockDim.x){float value=0;for(int j=0;j<=i;++j)value+=score[j]*V[(size_t(kh)*S+j)*D+d];out[(size_t(h)*S+i)*D+d]=value/z;}}
extern "C" void solve(const float*Q,const float*K,const float*V,const float*cos,const float*sin,float*output,int num_q_heads,int num_kv_heads,int seq_len,int head_dim){float *qr,*kr;size_t qb=size_t(num_q_heads)*seq_len*head_dim*sizeof(float),kb=size_t(num_kv_heads)*seq_len*head_dim*sizeof(float);cudaMalloc(&qr,qb);cudaMalloc(&kr,kb);int qn=num_q_heads*seq_len*head_dim,kn=num_kv_heads*seq_len*head_dim;rotate<<<(qn+255)/256,256>>>(Q,cos,sin,qr,num_q_heads,seq_len,head_dim);rotate<<<(kn+255)/256,256>>>(K,cos,sin,kr,num_kv_heads,seq_len,head_dim);size_t shared=size_t(seq_len+128)*sizeof(float);causal<<<num_q_heads*seq_len,128,shared>>>(qr,kr,V,output,num_q_heads,num_kv_heads,seq_len,head_dim);cudaDeviceSynchronize();cudaFree(kr);cudaFree(qr);}
```
