---
id: fused-logit-penalties
leetgpuId: 121
titleEn: Fused Logit Penalties
titleZh: 融合 Logit 惩罚
description: 汇总提示词与生成历史，并按固定顺序融合重复、频率和存在惩罚
problemUrl: https://leetgpu.com/challenges/fused-logit-penalties
updatedAt: 2026-11-12
---
## 题目定义

对每个请求和词元，先统计该词元是否出现在 prompt 或生成历史中，并单独统计它在 `output_tokens` 中的出现次数。`-1` 是任意位置都可能出现的 padding，不计数。

惩罚顺序固定：已见词元先执行 repetition，正 logit 除以 `r`，零或负 logit 乘以 `r`；再减 `frequency_penalty*生成次数`；生成过至少一次时再减一次 presence penalty。Prompt 中重复出现只影响“已见”状态，不增加 frequency 或 presence。

### 官方示例

```text
row0 logits=[1,-2,3,0.5,0,-1]
prompt=[1,3,-1], generated=[2,2,5,-1]
presence=0.5, frequency=0.25, repetition=2
output=[1,-4,0.5,0.25,0,-2.75]
```

完整两行官方输出为 `[[1,-4,0.5,0.25,0,-2.75],[2,1,-1,0,2.5,-3]]`。

### 完整约束

- `1 ≤ B ≤ 256`，`1 ≤ V ≤ 131072`
- `1 ≤ P ≤ 1024`，`1 ≤ G ≤ 512`
- token id 为 `-1` 或 `[0,V)`；logit 范围 `[-50,50]`
- presence/frequency 在 `[0,2]`，repetition 在 `[0.5,2]`
- logits、惩罚向量、output 为 FP32；token 数组为 int32
- 性能测试为 `B=256,V=128256,P=1024,G=512`
- 判定容差为 `1e-5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* logits, const int* prompt_tokens,
                      const int* output_tokens, const float* presence_penalty,
                      const float* frequency_penalty,
                      const float* repetition_penalty, float* output,
                      int B, int V, int P, int G) {}
```

输入 logits 必须保持不变。每个请求有独立的三个惩罚系数。

## 实现思路

分配 `B*V` 的 int32 计数表并清零。每个请求由唯一线程顺序扫描最多 1536 个历史位置：prompt 直接写入 seen 位，generated 同时累加 count；重复 token 会正确累加。

第二个 kernel 一线程处理一个 `(b,v)`，先根据 seen 应用有符号 repetition，再根据 count 依次应用 frequency 和 presence。词表阶段可并行覆盖最大约 3283 万个输出，避免逐 vocabulary entry 重扫历史。

## 参考实现

```cpp
#include <cuda_runtime.h>

__global__ void build_counts(const int* prompt,const int* generated,int* seen,int* count,int B,int V,int P,int G){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;for(int i=0;i<P;++i){int v=prompt[b*P+i];if(v>=0)seen[b*V+v]=1;}for(int i=0;i<G;++i){int v=generated[b*G+i];if(v>=0){seen[b*V+v]=1;count[b*V+v]++;}}}
__global__ void penalize(const float* logits,const int* seen,const int* count,const float* pp,const float* fp,const float* rp,float* out,long long total,int V){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;long long stride=(long long)gridDim.x*blockDim.x;for(;i<total;i+=stride){int b=(int)(i/V);float z=logits[i];if(seen[i])z=z>0?z/rp[b]:z*rp[b];int c=count[i];z-=fp[b]*c;if(c>0)z-=pp[b];out[i]=z;}}
extern "C" void solve(const float* logits,const int* prompt,const int* generated,const float* pp,const float* fp,const float* rp,float* output,int B,int V,int P,int G){int *seen,*count;size_t bytes=(size_t)B*V*sizeof(int);cudaMalloc(&seen,bytes);cudaMalloc(&count,bytes);cudaMemset(seen,0,bytes);cudaMemset(count,0,bytes);build_counts<<<(B+255)/256,256>>>(prompt,generated,seen,count,B,V,P,G);long long total=(long long)B*V;int blocks=(int)((total+255)/256);if(blocks>65535)blocks=65535;penalize<<<blocks,256>>>(logits,seen,count,pp,fp,rp,output,total,V);cudaFree(seen);cudaFree(count);cudaDeviceSynchronize();}
```
