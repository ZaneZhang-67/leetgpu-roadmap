---
id: hamming-top-k-search
leetgpuId: 122
titleEn: Hamming Top-K Search
titleZh: 汉明距离 Top-K 搜索
description: 对 256 位二进制编码并行计算汉明距离，并按距离与原始索引确定唯一 Top-K
problemUrl: https://leetgpu.com/challenges/hamming-top-k-search
updatedAt: 2026-11-12
---
## 题目定义

每个编码由 8 个 `int32` 字存储，共 256 位。编码 `i` 位于 `codes[8*i..8*i+7]`，距离为 8 个 XOR 结果的 popcount 之和，范围 `0..256`。有符号整数只作为存储类型，符号位也参与计数。

输出距离最小的 `k` 个原始索引，按 `(distance,index)` 升序排列。距离相同时，小索引优先，因此截止位置的 tie 也有唯一结果。

### 官方示例

```text
query=[1,2,3,0,0,0,0,0]
distances=[0,1,3,2,1], N=5, k=2
output=[0,1]
```

第二个官方示例的距离为 `[2,0,32,2]`、`k=2`，输出 `[1,0]`，索引 0 在距离 2 的 tie 中优先于索引 3。

### 完整约束

- `1 ≤ N ≤ 10000000`
- `1 ≤ k ≤ min(N,1024)`
- `codes` 有 `8*N` 个 int32，`query` 有 8 个 int32
- 32 位全部参与 XOR/popcount，包括符号位
- 性能测试使用 `N=4000000,k=128`
- 输出要求精确一致，容差为 0

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// codes, query, output are device pointers
extern "C" void solve(const int* codes, const int* query,
                      int* output, int N, int k) {}
```

输出只保存索引，不保存距离。参考实现额外保存 `N` 个 uint16 距离，以及每个 256 元素 Tile 的 257 个距离桶计数。

## 实现思路

汉明距离只有 `0..256` 共 257 个值，因此无需对 N 个元素全排序。第一阶段一线程计算一个编码的距离，同时累加全局桶和所属 256 元素 Tile 的桶计数。单 Block 对全局桶做前缀和，得到每个距离在最终有序序列中的起点。

第二阶段让 257 个线程分别沿 Tile 维计算同一距离的前缀计数。稳定写回时，每个 Tile Block 处理连续 256 个原索引；Shared Memory 记录各 Warp 的桶计数，`__match_any_sync` 给出 Warp 内同距离的先后排名。最终位置由“距离起点 + 更早 Tile 数量 + Tile 内更早索引数量”唯一确定，天然保持同距离小索引优先，不依赖跨 Block 自旋或调度顺序。

## 参考实现

```cpp
#include <cuda_runtime.h>
constexpr int BUCKETS=257,TILE=256,WARPS=8;
__global__ void make_dist(const int*c,const int*q,unsigned short*d,int*tile_count,int n){__shared__ int bucket[BUCKETS];int tid=(int)threadIdx.x,tile=(int)blockIdx.x,i=tile*TILE+tid;for(int x=tid;x<BUCKETS;x+=TILE)bucket[x]=0;__syncthreads();if(i<n){int value=0;for(int w=0;w<8;++w)value+=__popc((unsigned)c[(size_t)i*8+w]^(unsigned)q[w]);d[i]=(unsigned short)value;atomicAdd(bucket+value,1);}__syncthreads();for(int x=tid;x<BUCKETS;x+=TILE)tile_count[(size_t)tile*BUCKETS+x]=bucket[x];}
__global__ void tile_prefix(int*tile_count,int*hist,int tiles){int value=(int)threadIdx.x;if(value>=BUCKETS)return;int sum=0;for(int tile=0;tile<tiles;++tile){size_t at=(size_t)tile*BUCKETS+value;int count=tile_count[at];tile_count[at]=sum;sum+=count;}hist[value]=sum;}
__global__ void bucket_prefix(const int*hist,int*base){if(threadIdx.x==0){int sum=0;for(int value=0;value<BUCKETS;++value){base[value]=sum;sum+=hist[value];}}}
__global__ void stable_emit(const unsigned short*d,const int*base,const int*tile_before,int*out,int n,int k){__shared__ int warp_count[WARPS][BUCKETS];int tid=(int)threadIdx.x,tile=(int)blockIdx.x,i=tile*TILE+tid,lane=tid&31,warp=tid>>5;for(int x=tid;x<WARPS*BUCKETS;x+=TILE)((int*)warp_count)[x]=0;__syncthreads();int value=i<n?(int)d[i]:BUCKETS;if(i<n)atomicAdd(&warp_count[warp][value],1);__syncthreads();unsigned same=__match_any_sync(0xffffffffu,value);if(i<n){int rank=__popc(same&((1u<<lane)-1u));for(int w=0;w<warp;++w)rank+=warp_count[w][value];int pos=base[value]+tile_before[(size_t)tile*BUCKETS+value]+rank;if(pos<k)out[pos]=i;}}
extern "C" void solve(const int*c,const int*q,int*out,int n,int k){int tiles=(n+TILE-1)/TILE;unsigned short*d;int *tile_count,*hist,*base;cudaMalloc(&d,(size_t)n*sizeof(unsigned short));cudaMalloc(&tile_count,(size_t)tiles*BUCKETS*sizeof(int));cudaMalloc(&hist,BUCKETS*sizeof(int));cudaMalloc(&base,BUCKETS*sizeof(int));make_dist<<<tiles,TILE>>>(c,q,d,tile_count,n);tile_prefix<<<1,512>>>(tile_count,hist,tiles);bucket_prefix<<<1,1>>>(hist,base);stable_emit<<<tiles,TILE>>>(d,base,tile_count,out,n,k);cudaDeviceSynchronize();cudaFree(base);cudaFree(hist);cudaFree(tile_count);cudaFree(d);}
```
