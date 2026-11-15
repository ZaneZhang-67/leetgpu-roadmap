---
id: diffusion-transformer-block
leetgpuId: 116
titleEn: Diffusion Transformer Block
titleZh: Diffusion Transformer Block
description: 使用 CUDA 实现带 adaLN 调制的双向自注意力与 GELU MLP 残差块
problemUrl: https://leetgpu.com/challenges/diffusion-transformer-block
updatedAt: 2026-11-12
---
## 题目定义

固定 `D=512`、8 heads、head_dim=64、MLP=2048。先由 `SiLU(c)W_adaᵀ+b` 生成六组 `(shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp)`。两条子层均使用无仿射 LayerNorm、`eps=1e-6`，调制为 `norm*(1+scale)+shift`。

注意力为无 causal mask 的双向注意力，输出经 `W_o` 后以 `gate_msa` 加到 x。MLP 使用 tanh 近似 GELU，经 `W_fc2` 后以 `gate_mlp` 加到第一条残差。

### 示例

官方示例形状：`x(1,4,512)`、`c(1,512)`、packed weights 长度 4,726,272，输出 `(1,4,512)`。所有张量均为 FP32，`1 ≤ batch_size ≤ 16`、`1 ≤ seq_len ≤ 4096`。性能用例为 batch=4、seq=1024，容差 `1e-3`。

## 模板代码解析

签名为：

```text
extern "C" void solve(const float* x, const float* c, float* output,
                      const float* weights, int batch_size, int seq_len)
```

packed 顺序依次是 Wada/bada、Wqkv/bqkv、Wo/bo、Wfc1/bfc1、Wfc2/bfc2；矩阵均为 `(out,in)` 行主序。关键偏移为 0、1572864、1575936、2362368、2363904、2626048、2626560、3675136、3677184、4725760。

## 实现思路

参考实现采用 16×16 Shared Memory tiled GEMM 与逐 Token kernel，分配 mod、h、qkv、attention、x1、MLP 临时缓冲。注意力为每个 `(batch,head,query)` 一个 Block，在线稳定 Softmax 扫描 key，避免 `S×S` 分数矩阵。

同一 tiled kernel 覆盖 adaLN、QKV、输出投影和两层 MLP，边缘 Tile 补零。其余 kernel 固定 packed layout、两遍中心方差、调制、无 mask、残差门控和 tanh GELU 的数值顺序。

## 参考实现

```cpp
#include <cmath>
#include <cuda_runtime.h>

constexpr int D=512, HEADS=8, DH=64, FF=2048;
__global__ void linear(const float* a,const float* w,const float* bias,float* o,int rows,int in,int out){
 __shared__ float at[16][16],wt[16][16];int tx=threadIdx.x,ty=threadIdx.y,j=blockIdx.x*16+tx,i=blockIdx.y*16+ty;float sum=0;
 for(int k0=0;k0<in;k0+=16){int ka=k0+tx,kw=k0+ty;at[ty][tx]=(i<rows&&ka<in)?a[(size_t)i*in+ka]:0;wt[ty][tx]=(kw<in&&j<out)?w[(size_t)j*in+kw]:0;__syncthreads();for(int k=0;k<16;++k)sum+=at[ty][k]*wt[k][tx];__syncthreads();}
 if(i<rows&&j<out)o[i*out+j]=sum+(bias?bias[j]:0);
}
__global__ void silu_kernel(const float* c,float* z,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)z[i]=c[i]/(1+expf(-c[i]));}
__global__ void norm_mod(const float*x,const float*mod,float*h,int rows,int tokens,int which){
 int r=blockIdx.x,t=threadIdx.x;if(r>=rows)return;__shared__ float sm[256];float s=0;for(int d=t;d<D;d+=256)s+=x[r*D+d];sm[t]=s;__syncthreads();for(int o=128;o;o>>=1){if(t<o)sm[t]+=sm[t+o];__syncthreads();}float mean=sm[0]/D,ss=0;for(int d=t;d<D;d+=256){float z=x[r*D+d]-mean;ss+=z*z;}sm[t]=ss;__syncthreads();for(int o=128;o;o>>=1){if(t<o)sm[t]+=sm[t+o];__syncthreads();}float inv=rsqrtf(sm[0]/D+1e-6f);int b=r/tokens;for(int d=t;d<D;d+=256){int base=b*6*D+(which?3:0)*D;h[r*D+d]=(x[r*D+d]-mean)*inv*(1+mod[base+D+d])+mod[base+d];}}
__global__ void attention(const float*qkv,float*out,int S){int q=blockIdx.x,h=blockIdx.y,b=blockIdx.z,t=threadIdx.x;__shared__ float acc[64],red[256];for(int d=t;d<DH;d+=256)acc[d]=0;__shared__ float m,l,of,nw;if(t==0){m=-1e30f;l=0;}__syncthreads();for(int k=0;k<S;++k){float p=0;for(int d=t;d<DH;d+=256)p+=qkv[(b*S+q)*3*D+h*DH+d]*qkv[(b*S+k)*3*D+D+h*DH+d];red[t]=p;__syncthreads();for(int o=128;o;o>>=1){if(t<o)red[t]+=red[t+o];__syncthreads();}if(t==0){float s=red[0]*0.125f,nm=fmaxf(m,s);of=expf(m-nm);nw=expf(s-nm);l=l*of+nw;m=nm;}__syncthreads();for(int d=t;d<DH;d+=256)acc[d]=acc[d]*of+nw*qkv[(b*S+k)*3*D+2*D+h*DH+d];__syncthreads();}for(int d=t;d<DH;d+=256)out[(b*S+q)*D+h*DH+d]=acc[d]/l;}
__global__ void residual(float*x,const float*y,const float*mod,float*o,int rows,int tokens,int gate){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*D){int b=(i/D)/tokens,d=i%D;o[i]=x[i]+mod[b*6*D+gate*D+d]*y[i];}}
__global__ void gelu(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=.5f*v*(1+tanhf(.7978845608f*(v+.044715f*v*v*v)));}}
extern "C" void solve(const float*x,const float*c,float*output,const float*w,int B,int S){
 int R=B*S;float *cs,*mod,*h,*qkv,*att,*proj,*x1,*ff;cudaMalloc(&cs,B*D*4);cudaMalloc(&mod,B*6*D*4);cudaMalloc(&h,R*D*4);cudaMalloc(&qkv,R*3*D*4);cudaMalloc(&att,R*D*4);cudaMalloc(&proj,R*D*4);cudaMalloc(&x1,R*D*4);cudaMalloc(&ff,R*FF*4);
 dim3 th(16,16);silu_kernel<<<(B*D+255)/256,256>>>(c,cs,B*D);linear<<<dim3((6*D+15)/16,(B+15)/16),th>>>(cs,w,w+1572864,mod,B,D,6*D);norm_mod<<<R,256>>>(x,mod,h,R,S,0);linear<<<dim3((3*D+15)/16,(R+15)/16),th>>>(h,w+1575936,w+2362368,qkv,R,D,3*D);attention<<<dim3(S,HEADS,B),256>>>(qkv,att,S);linear<<<dim3(32,(R+15)/16),th>>>(att,w+2363904,w+2626048,proj,R,D,D);residual<<<(R*D+255)/256,256>>>(const_cast<float*>(x),proj,mod,x1,R,S,2);norm_mod<<<R,256>>>(x1,mod,h,R,S,1);linear<<<dim3(128,(R+15)/16),th>>>(h,w+2626560,w+3675136,ff,R,D,FF);gelu<<<(R*FF+255)/256,256>>>(ff,R*FF);linear<<<dim3(32,(R+15)/16),th>>>(ff,w+3677184,w+4725760,proj,R,FF,D);residual<<<(R*D+255)/256,256>>>(x1,proj,mod,output,R,S,5);cudaDeviceSynchronize();cudaFree(ff);cudaFree(x1);cudaFree(proj);cudaFree(att);cudaFree(qkv);cudaFree(h);cudaFree(mod);cudaFree(cs);
}
```
