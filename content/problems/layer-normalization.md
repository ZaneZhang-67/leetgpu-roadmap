---
id: layer-normalization
leetgpuId: 113
titleEn: Layer Normalization
titleZh: 层归一化
description: 使用 CUDA 按行计算稳定均值与总体方差，并应用逐特征仿射变换
problemUrl: https://leetgpu.com/challenges/layer-normalization
updatedAt: 2026-11-12
---
## 题目定义

对 `(N,C)` FP32 输入的每一行独立归一化：

```text
mean[i] = Σ input[i,j] / C
var[i]  = Σ (input[i,j]-mean[i])² / C
output[i,j] = weight[j]*(input[i,j]-mean[i])/sqrt(var[i]+eps)+bias[j]
```

方差是总体方差，weight 和 bias 长度均为 `C`，在所有行间共享。

### 示例

```text
input  = [[1,2,3,4],[-1,0,0,1]]
weight = [1,1,1,1], bias=[0,0,0,0], eps=1e-5
output = [[-1.3416,-0.4472,0.4472,1.3416],
          [-1.4142,0,0,1.4142]]
```

**约束与测试**

- `1≤N≤65,536`，`1≤C≤4096`，`eps=1e-5`
- 输入 `[-100,100]`，weight `[0.1,10]`，bias `[-10,10]`
- 测试覆盖 `C=1`、全零、负数、`C=4096` 和非二次幂 `C=30/100/255`
- 性能测试 `N=65,536,C=512`，容差 `atol=rtol=1e-4`

## 模板代码解析

```cpp
#include <cuda_runtime.h>

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {}
```

官方 starter 只提供空入口。输入输出按行主序存储，需要自行实现按行统计、仿射变换和启动配置。

## 实现思路

均值与方差分别启动归约 kernel，保持官方 runtime 的两次独立统计。启动配置按输入长度与输出行数选择最多 512 个线程；连续长行的均值使用四元素向量载入和四路局部累加，方差使用两元素向量载入及 FP32 Welford `(count,mean,M2)`，随后按相同二叉树拓扑合并。未对齐行首、尾元素和同一 Block 内的多行也分别处理。

总体方差为 `M2/C`。第三个 kernel 按 FP32 顺序执行 `sqrt(var+eps)`、减法、除法和仿射变换。它避免 `E[x²]-E[x]²` 的抵消，也避免用更高精度数学结果替代 runtime 在大偏置低方差输入上的规定舍入路径。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct Welford { float mean, m2, nf; int n; };
__device__ Welford add_value(Welford a, float x) {
    int n = a.n + 1; float nf = (float)n, delta = x - a.mean;
    float mean = a.mean + delta / nf;
    return {mean, a.m2 + delta * (x - mean), nf, n};
}
__device__ Welford combine(Welford a, Welford b) {
    if (a.nf == 0.0f) return b; if (b.nf == 0.0f) return a;
    float delta = b.mean - a.mean, count = a.nf + b.nf;
    float ratio = b.nf / count;
    return {a.mean + delta * ratio,
            a.m2 + b.m2 + delta * delta * a.nf * ratio,
            count, -1};
}

__global__ void runtime_mean(const float* input, float* means, int N, int C,
                             int vectorized, int split_y) {
    __shared__ float shared[512];
    int tx=(int)threadIdx.x, ty=(int)threadIdx.y;
    int bw=(int)blockDim.x, bh=(int)blockDim.y;
    int row=split_y?(int)blockIdx.x:(int)blockIdx.x*bh+ty;
    int slot=ty*bw+tx, start=tx+(split_y?ty*bw:0);
    int stride=bw*(split_y?bh:1); float value=0.0f;
    if(row<N&&vectorized){
        const float*data=input+(size_t)row*C; int end=C;
        int shift=(int)(((uintptr_t)data&15u)/sizeof(float));
        float part[4]={0,0,0,0};
        if(shift){data-=shift;end+=shift;if((!split_y||ty==0)&&tx>=shift&&tx<4)part[0]+=data[tx];end-=4;data+=4;shift=4-shift;}
        for(int idx=start;idx*4+3<end;idx+=stride){float4 v=*reinterpret_cast<const float4*>(data+(size_t)idx*4);part[0]+=v.x;part[1]+=v.y;part[2]+=v.z;part[3]+=v.w;}
        if((!split_y||ty==0)){int tail=end-end%4+tx;if(tail<end)part[0]+=data[tail];}
        value=((part[0]+part[1])+part[2])+part[3];
    }else if(row<N){
        float part[4]={0,0,0,0};int idx=start;
        while(idx+3*stride<C){part[0]+=input[(size_t)row*C+idx];part[1]+=input[(size_t)row*C+idx+stride];part[2]+=input[(size_t)row*C+idx+2*stride];part[3]+=input[(size_t)row*C+idx+3*stride];idx+=4*stride;}
        for(int j=0;j<4&&idx<C;++j,idx+=stride)part[j]+=input[(size_t)row*C+idx];
        value=((part[0]+part[1])+part[2])+part[3];
    }
    shared[slot]=value;__syncthreads();
    for(int off=bw/2;off;off>>=1){if(tx<off)shared[slot]+=shared[slot+off];__syncthreads();}
    if(split_y)for(int off=bh/2;off;off>>=1){if(tx==0&&ty<off)shared[slot]+=shared[slot+off*bw];__syncthreads();}
    if(row<N&&tx==0&&(!split_y||ty==0))means[row]=shared[slot]*((float)N/(float)((size_t)N*C));
}

__global__ void runtime_variance(const float* input, float* vars, int N, int C,
                                 int vectorized, int split_y) {
    __shared__ Welford shared[512];
    int tx=(int)threadIdx.x,ty=(int)threadIdx.y,bw=(int)blockDim.x,bh=(int)blockDim.y;
    int row=split_y?(int)blockIdx.x:(int)blockIdx.x*bh+ty;
    int slot=ty*bw+tx,start=tx+(split_y?ty*bw:0),stride=bw*(split_y?bh:1);
    Welford value={0,0,0,0};
    if(row<N&&vectorized){
        const float*data=input+(size_t)row*C;int end=C,shift=(int)(((uintptr_t)data&7u)/sizeof(float));Welford part[2]={{0,0,0,0},{0,0,0,0}};
        if(shift){data-=shift;end+=shift;if((!split_y||ty==0)&&tx>=shift&&tx<2)part[0]=add_value(part[0],data[tx]);end-=2;data+=2;shift=2-shift;}
        for(int idx=start;idx*2+1<end;idx+=stride){float2 v=*reinterpret_cast<const float2*>(data+(size_t)idx*2);part[0]=add_value(part[0],v.x);part[1]=add_value(part[1],v.y);}
        if((!split_y||ty==0)){int tail=end-end%2+tx;if(tail<end)part[0]=add_value(part[0],data[tail]);}
        value=combine(part[0],part[1]);
    }else if(row<N){
        Welford part[2]={{0,0,0,0},{0,0,0,0}};int idx=start;
        while(idx+stride<C){part[0]=add_value(part[0],input[(size_t)row*C+idx]);part[1]=add_value(part[1],input[(size_t)row*C+idx+stride]);idx+=2*stride;}
        if(idx<C)part[0]=add_value(part[0],input[(size_t)row*C+idx]);value=combine(part[0],part[1]);
    }
    shared[slot]=value;__syncthreads();
    for(int off=bw/2;off;off>>=1){if(tx<off)shared[slot]=combine(shared[slot],shared[slot+off]);__syncthreads();}
    if(split_y)for(int off=bh/2;off;off>>=1){if(tx==0&&ty<off)shared[slot]=combine(shared[slot],shared[slot+off*bw]);__syncthreads();}
    if(row<N&&tx==0&&(!split_y||ty==0))vars[row]=shared[slot].m2/shared[slot].nf;
}

__global__ void affine(const float*input,const float*weight,const float*bias,
                       const float*mean,const float*var,float*out,size_t total,
                       int C,float eps){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<total){int c=(int)(i%C);float centered=__fsub_rn(input[i],mean[i/C]);float sd=sqrtf(__fadd_rn(var[i/C],eps));float normalized=__fdiv_rn(centered,sd);out[i]=__fadd_rn(__fmul_rn(weight[c],normalized),bias[c]);}}

__host__ int floor_pow2(int x){int p=1;while(p<=x/2)p<<=1;return p;}
__host__ void reduction_config(int C,int N,int vec,int&bw,int&bh,int&split){int d0=C/vec;if(d0<1)d0=1;int p0=d0<512?floor_pow2(d0):512,p1=N<512?floor_pow2(N):512;bw=p0<32?p0:32;bh=p1<512/bw?p1:512/bw;bw=p0<512/bh?p0:512/bh;int values=(C+bw-1)/bw,threshold=bh*16<256?bh*16:256;split=values>=threshold;}
extern "C" void solve(const float*input,const float*weight,const float*bias,float*out,int N,int C,float eps){float *mean,*var;cudaMalloc(&mean,(size_t)N*sizeof(float));cudaMalloc(&var,(size_t)N*sizeof(float));int bw,bh,split;reduction_config(C,N,C>=128?4:1,bw,bh,split);runtime_mean<<<split?N:(N+bh-1)/bh,dim3(bw,bh)>>>(input,mean,N,C,C>=128,split);reduction_config(C,N,C>=128?2:1,bw,bh,split);runtime_variance<<<split?N:(N+bh-1)/bh,dim3(bw,bh)>>>(input,var,N,C,C>=128,split);size_t total=(size_t)N*C;affine<<<(total+255)/256,256>>>(input,weight,bias,mean,var,out,total,C,eps);cudaDeviceSynchronize();cudaFree(var);cudaFree(mean);}
```
