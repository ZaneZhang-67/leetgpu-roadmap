---
id: softcapped-sliding-window-attention
leetgpuId: 152
titleEn: Softcapped Sliding Window Attention
titleZh: Softcap 滑动窗口注意力
description: 使用 CUDA 实现带 GQA、因果滑动窗口及 score softcap 的稳定注意力
problemUrl: https://leetgpu.com/challenges/softcapped-sliding-window-attention
updatedAt: 2026-11-12
---
## 题目定义

布局为 `Q:[Hq,S,D]`、`K/V:[Hkv,S,D]`、`output:[Hq,S,D]`。查询头 h 映射到 `kv_head=h/(Hq/Hkv)`。查询位置 i 只访问：

```text
max(0,i-window_size+1) ≤ j ≤ i
```

先缩放原始点积，再做 softcap：

```text
raw=dot(Q,K)/√D
score=softcap×tanh(raw/softcap)
```

最后仅在可见窗口内计算 Softmax 与 V 加权和。softcap 不能放到缩放之前。

### 实现要求与约束

- 所有张量均为 FP32
- `1≤Hkv≤Hq≤32`，且 Hq 能被 Hkv 整除
- `1≤S≤4096`，`1≤D≤128`
- `1≤window_size≤8192`，允许大于 S
- `0<softcap≤50`
- 每个输出元素都必须写入
- 性能测试 `Hq=16,Hkv=4,S=4096,D=64,window=1024,softcap=10`
- 绝对误差和相对误差均为 `1e-5`

官方示例 `Hq=2,Hkv=1,S=3,D=2,window=2,softcap=1`；第 2 行只访问键 1、2。

## 模板代码解析

```cpp
#include <cuda_runtime.h>
extern "C" void solve(const float* Q,const float* K,const float* V,float* output,int num_q_heads,int num_kv_heads,int seq_len,int head_dim,int window_size,float softcap) {}
```

## 实现思路

一个 128 线程 Block 负责 `(query_head,position)`。对窗口中每个 key，所有 lane 协作计算一次 Q·K，并保存 softcap score。窗口最大 4096，因此动态 Shared Memory 最多 16 KiB。随后 128 lane 归约 score 最大值；计算指数和时把未归一化权重回写 score，之后由 lane 分担 D 个输出维并复用这些权重。点积不会因每个输出维重复计算，复杂度为 `O(Hq×S×window×D)`。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
__global__ void softcap_attention(const float*Q,const float*K,const float*V,float*out,int Hq,int Hkv,int S,int D,int window,float cap){extern __shared__ float sm[];float*score=sm;float*reduce=sm+min(window,S);int x=blockIdx.x,lane=threadIdx.x,h=x/S,i=x%S,group=Hq/Hkv,kh=h/group,start=max(0,i-window+1),count=i-start+1;float scale=rsqrtf(float(D));for(int p=0;p<count;++p){int j=start+p;float part=lane<D?Q[(size_t(h)*S+i)*D+lane]*K[(size_t(kh)*S+j)*D+lane]:0.0f;reduce[lane]=part;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)reduce[lane]+=reduce[lane+r];__syncthreads();}if(lane==0){float raw=reduce[0]*scale;score[p]=cap*tanhf(raw/cap);}__syncthreads();}float m=-CUDART_INF_F;for(int p=lane;p<count;p+=blockDim.x)m=fmaxf(m,score[p]);reduce[lane]=m;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)reduce[lane]=fmaxf(reduce[lane],reduce[lane+r]);__syncthreads();}m=reduce[0];float z=0;for(int p=lane;p<count;p+=blockDim.x){score[p]=expf(score[p]-m);z+=score[p];}reduce[lane]=z;__syncthreads();for(int r=64;r;r>>=1){if(lane<r)reduce[lane]+=reduce[lane+r];__syncthreads();}z=reduce[0];for(int d=lane;d<D;d+=blockDim.x){float value=0;for(int p=0;p<count;++p)value+=score[p]*V[(size_t(kh)*S+start+p)*D+d];out[(size_t(h)*S+i)*D+d]=value/z;}}
extern "C" void solve(const float* Q,const float* K,const float* V,float* output,int num_q_heads,int num_kv_heads,int seq_len,int head_dim,int window_size,float softcap){int visible=min(window_size,seq_len);size_t shared=size_t(visible+128)*sizeof(float);softcap_attention<<<num_q_heads*seq_len,128,shared>>>(Q,K,V,output,num_q_heads,num_kv_heads,seq_len,head_dim,window_size,softcap);cudaDeviceSynchronize();}
```
