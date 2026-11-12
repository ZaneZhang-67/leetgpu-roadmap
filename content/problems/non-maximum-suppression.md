---
id: non-maximum-suppression
leetgpuId: 117
titleEn: Non-Maximum Suppression
titleZh: 非极大值抑制
description: 按稳定分数顺序执行贪心框抑制，并把存活标记写回原始索引
problemUrl: https://leetgpu.com/challenges/non-maximum-suppression
updatedAt: 2026-11-12
---
## 题目定义

输入 `N` 个 `[x1,y1,x2,y2]` 框及分数。先按分数降序排列；分数相同时，原始索引小者优先。依序访问，尚未被抑制的框写 `keep=1`，并抑制后续所有与它的 IoU **严格大于**阈值的框。已被抑制的框不会再抑制其他框。

`keep[N]` 按原始框顺序写入，每项必须是 0 或 1，而不是紧凑的存活索引列表。

### 官方示例

```text
scores=[0.90,0.75,0.95,0.60,0.55,0.50]
iou_threshold=0.5
keep=[1,0,1,0,1,0]
```

### 完整约束

- `1 ≤ N ≤ 20000`
- boxes 为 FP32 `[N,4]` 行主序，满足 `x1<x2,y1<y2`
- 坐标为 `[-10000,10000]` 内的整数值；分数在 `[0,1]`
- `0 ≤ iou_threshold ≤ 1`；只有 `IoU>threshold` 才抑制
- keep 为 int32；判定容差为 0
- 性能测试使用 `N=16384,threshold=0.5`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// boxes, scores, keep are device pointers
extern "C" void solve(const float* boxes, const float* scores,
                      int* keep, int N, float iou_threshold) {}
```

稳定排序和贪心链式依赖都需要自行实现。参考实现先在 GPU 上按 `(-score,index)` 归并排序，再让一个 kernel 按排序结果执行严格的贪心语义。

## 实现思路

先用并行归并排序得到稳定的 score 顺序。随后构造标准 NMS bitmask：每个二维 tile 比较 64 个行框和 64 个列框，列框位于排序后更晚位置且 `IoU>threshold` 时，在对应行的 uint64 mask 中置位。这里仍严格使用原始索引 tie 和严格大于阈值。

消去阶段只顺序扫描 N 个 rank，但不再逐对计算 IoU。它维护长度 `ceil(N/64)` 的 removed bitset；存活框只需把预计算 mask 按 64 位字 OR 进去。计算密集的 `O(N²)` IoU 已由大量 tile Block 并行完成，顺序部分为 `O(N²/64)` 位运算，并按原索引写 keep。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <stdint.h>
__device__ bool prior(float sa,int ia,float sb,int ib){return sa>sb||(sa==sb&&ia<ib);}
__global__ void init(int*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=i;}
__global__ void merge(const float*s,const int*in,int*out,int n,int w){int l=blockIdx.x*2*w,m=min(l+w,n),e=min(l+2*w,n);if(threadIdx.x==0){int a=l,b=m,p=l;while(a<m&&b<e){int ia=in[a],ib=in[b];out[p++]=prior(s[ia],ia,s[ib],ib)?in[a++]:in[b++];}while(a<m)out[p++]=in[a++];while(b<e)out[p++]=in[b++];}}
__device__ float overlap(const float*b,int a,int c){float x1=fmaxf(b[a*4],b[c*4]),y1=fmaxf(b[a*4+1],b[c*4+1]),x2=fminf(b[a*4+2],b[c*4+2]),y2=fminf(b[a*4+3],b[c*4+3]);float w=fmaxf(0.f,x2-x1),h=fmaxf(0.f,y2-y1),i=w*h,aa=(b[a*4+2]-b[a*4])*(b[a*4+3]-b[a*4+1]),ac=(b[c*4+2]-b[c*4])*(b[c*4+3]-b[c*4+1]);return i/(aa+ac-i);}
__global__ void masks(const float*b,const int*o,unsigned long long*mask,int n,int words,float th){int row=blockIdx.y*64+threadIdx.x,colblock=blockIdx.x;if(row>=n)return;unsigned long long bits=0;int begin=colblock*64;for(int j=0;j<64&&begin+j<n;++j){int col=begin+j;if(col>row&&overlap(b,o[row],o[col])>th)bits|=1ULL<<j;}mask[(size_t)row*words+colblock]=bits;}
__global__ void consume(const int*o,const unsigned long long*mask,unsigned long long*removed,int*keep,int n,int words){if(blockIdx.x||threadIdx.x)return;for(int i=0;i<n;++i)keep[i]=0;for(int r=0;r<n;++r){if((removed[r>>6]>>(r&63))&1ULL)continue;keep[o[r]]=1;const unsigned long long*src=mask+(size_t)r*words;for(int w=r>>6;w<words;++w)removed[w]|=src[w];}}
extern "C" void solve(const float*b,const float*s,int*keep,int n,float th){int *a,*c;cudaMalloc(&a,(size_t)n*4);cudaMalloc(&c,(size_t)n*4);init<<<(n+255)/256,256>>>(a,n);for(int w=1;w<n;w*=2){merge<<<(n+2*w-1)/(2*w),1>>>(s,a,c,n,w);int*t=a;a=c;c=t;if(w>n/2)break;}int words=(n+63)/64;unsigned long long *mask,*removed;cudaMalloc(&mask,(size_t)n*words*8);cudaMalloc(&removed,(size_t)words*8);cudaMemset(mask,0,(size_t)n*words*8);cudaMemset(removed,0,(size_t)words*8);masks<<<dim3(words,(n+63)/64),64>>>(b,a,mask,n,words,th);consume<<<1,1>>>(a,mask,removed,keep,n,words);cudaDeviceSynchronize();cudaFree(mask);cudaFree(removed);cudaFree(a);cudaFree(c);}
```
