---
id: fast-fourier-transform
leetgpuId: 39
titleEn: Fast Fourier Transform
titleZh: 快速傅里叶变换
description: 使用 CUDA 计算任意合法长度的一维复数 DFT，结合 radix-2 FFT 与 Bluestein 卷积保持 O(N log N) 复杂度
problemUrl: https://leetgpu.com/challenges/fast-fourier-transform
updatedAt: 2026-11-12
---
## 题目定义

给定包含 `N` 个复数的一维信号 `signal`，计算正向离散傅里叶变换并写入 `spectrum`：

```text
X[k] = Σ x[n] × exp(-j × 2πkn / N)
```

其中 `0 ≤ k < N`、`0 ≤ n < N`

输入和输出都使用实部、虚部交错布局：

```text
[real0, imag0, real1, imag1, ...]
```

数组长度均为 `2 × N`

### 实现要求

- 不允许使用 cuFFT 等外部库
- `solve` 函数签名必须保持不变
- 最终结果必须写入 `spectrum`
- 变换过程必须在 GPU 上执行，不调用主机端 FFT

### 示例 1

```text
N = 4
signal = [1, 0, 0, 0, 0, 0, 0, 0]
spectrum = [1, 0, 1, 0, 1, 0, 1, 0]
```

### 示例 2

```text
N = 2
signal = [1, 0, 1, 0]
spectrum = [2, 0, 0, 0]
```

### 约束

- `1 ≤ N ≤ 262,144`
- 输入和输出值均为 32 位浮点数
- 输入和输出数组长度均为 `2 × N`
- 绝对误差不超过 `1e-3`，相对误差不超过 `1e-3`
- reference 使用 `torch.fft.fft`，功能测试明确包含非 2 的幂长度 `N = 250`
- 性能测试使用 `N = 262,144`

这是未归一化的正向变换：指数符号为负，输出不除以 `N`

## 模板代码解析

平台提供固定的 CUDA 入口：

```cpp
#include <cuda_runtime.h>

// signal and spectrum are device pointers
extern "C" void solve(
    const float* signal,
    float* spectrum,
    int N
) {}
```

- `signal` 是 GPU 上长度为 `2 × N` 的交错复数输入
- `spectrum` 是同布局的输出
- `N` 表示复数样本数，而不是浮点数数量

模板没有限制 `N` 为 2 的幂，也没有提供临时缓冲区或 kernel

## 实现思路

reference 接受任意允许的 `N`，因此只写 radix-2 FFT 会在 `N = 250` 等测试上出错参考实现分两条路径：2 的幂长度直接使用 radix-2 FFT，其他长度通过 Bluestein 变换归约到更大的 2 的幂卷积

### 1. 复数布局

第 `i` 个复数的实部和虚部分别位于：

```text
real = array[2 × i]
imag = array[2 × i + 1]
```

复数乘法 `(a + jb)(c + jd)` 展开为：

```text
real = ac - bd
imag = ad + bc
```

### 2. 2 的幂路径：位逆序与蝶形

迭代 Cooley-Tukey radix-2 FFT 先将输入按 `log2(N)` 位逆序复制到工作数组随后按跨度 `length = 2, 4, 8, ... N` 执行蝶形

每个蝶形使用旋转因子：

```text
W = exp(-j × 2π × offset / length)
```

负号保证方向与 `torch.fft.fft` 一致每一层由独立 kernel 执行，kernel 启动顺序提供全局阶段同步

### 3. 非 2 的幂路径

Bluestein 恒等式把 DFT 写成 chirp 卷积：

```text
a[n] = x[n] × exp(-jπn²/N)
b[m] = exp(+jπm²/N)
X[k] = exp(-jπk²/N) × (a * b)[k]
```

取 `M` 为不小于 `2N-1` 的最小 2 的幂，把正负下标的 `b` 映射到长度 `M` 的循环数组卷积由两次长度 `M` 的正向 FFT、逐点复乘和一次逆向 FFT 完成，复杂度保持 `O(N log N)`，最大合法长度只需 `M = 524,288`

逆向 FFT 使用共轭恒等式 `IFFT(z) = conj(FFT(conj(z))) / M`，因此可以复用同一个 radix-2 kernel

### 4. 方向和归一化

正向变换的角度为负：

```cpp
angle = -2πkn / N;
```

结果不除以 `N`只有逆变换通常才进行 `1/N` 归一化，本题没有逆变换参数

### 5. 数值输出

FFT 工作数组与输出使用 32 位浮点数，以匹配 starter 和 reference 数据类型Bluestein 的 `n²` 相位先在 64 位整数中对 `2N` 取模，再用双精度 `sincos` 生成 chirp，避免大下标相乘和巨大角度约简带来的相位误差

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

constexpr float kTwoPi = 6.28318530717958647692f;

__device__ unsigned int reverse_bits(
    unsigned int value,
    int bits
) {
    unsigned int reversed = 0;
    for (int i = 0; i < bits; ++i) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

__global__ void bit_reverse_copy(
    const float* signal,
    float* data,
    int N,
    int bits
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    unsigned int destination = reverse_bits(
        static_cast<unsigned int>(index), bits
    );
    data[2 * destination] = signal[2 * index];
    data[2 * destination + 1] = signal[2 * index + 1];
}

__global__ void fft_stage(
    float* data,
    int N,
    int length
) {
    int butterfly = blockIdx.x * blockDim.x + threadIdx.x;
    if (butterfly >= N / 2) return;

    int half = length / 2;
    int group = butterfly / half;
    int offset = butterfly % half;
    int left = group * length + offset;
    int right = left + half;

    float angle = -kTwoPi * static_cast<float>(offset)
        / static_cast<float>(length);
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);

    float rightReal = data[2 * right];
    float rightImag = data[2 * right + 1];
    float twiddledReal = cosine * rightReal - sine * rightImag;
    float twiddledImag = sine * rightReal + cosine * rightImag;

    float leftReal = data[2 * left];
    float leftImag = data[2 * left + 1];

    data[2 * left] = leftReal + twiddledReal;
    data[2 * left + 1] = leftImag + twiddledImag;
    data[2 * right] = leftReal - twiddledReal;
    data[2 * right + 1] = leftImag - twiddledImag;
}

__global__ void bit_reverse_conjugate_copy(
    const float* source,
    float* destination,
    int N,
    int bits
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    unsigned int reversed = reverse_bits(
        static_cast<unsigned int>(index), bits
    );
    destination[2 * reversed] = source[2 * index];
    destination[2 * reversed + 1] = -source[2 * index + 1];
}

__global__ void initialize_bluestein(
    const float* signal,
    float* a,
    float* b,
    int N,
    int M
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    unsigned long long square =
        static_cast<unsigned long long>(n) * static_cast<unsigned long long>(n);
    unsigned long long phase = square % (2ull * static_cast<unsigned long long>(N));
    double angle = 3.14159265358979323846 * static_cast<double>(phase) / N;
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);

    float chirpReal = static_cast<float>(cosine);
    float chirpImag = static_cast<float>(sine);
    float inputReal = signal[2 * n];
    float inputImag = signal[2 * n + 1];

    a[2 * n] = inputReal * chirpReal + inputImag * chirpImag;
    a[2 * n + 1] = inputImag * chirpReal - inputReal * chirpImag;
    b[2 * n] = chirpReal;
    b[2 * n + 1] = chirpImag;
    if (n > 0) {
        int negative = M - n;
        b[2 * negative] = chirpReal;
        b[2 * negative + 1] = chirpImag;
    }
}

__global__ void multiply_complex(
    const float* left,
    const float* right,
    float* output,
    int N
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float leftReal = left[2 * index];
    float leftImag = left[2 * index + 1];
    float rightReal = right[2 * index];
    float rightImag = right[2 * index + 1];
    output[2 * index] = leftReal * rightReal - leftImag * rightImag;
    output[2 * index + 1] = leftReal * rightImag + leftImag * rightReal;
}

__global__ void finish_bluestein(
    const float* inverseWork,
    float* spectrum,
    int N,
    int M
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    unsigned long long square =
        static_cast<unsigned long long>(k) * static_cast<unsigned long long>(k);
    unsigned long long phase = square % (2ull * static_cast<unsigned long long>(N));
    double angle = 3.14159265358979323846 * static_cast<double>(phase) / N;
    double sine;
    double cosine;
    sincos(angle, &sine, &cosine);

    float valueReal = inverseWork[2 * k] / M;
    float valueImag = -inverseWork[2 * k + 1] / M;
    float chirpReal = static_cast<float>(cosine);
    float chirpImag = static_cast<float>(sine);
    spectrum[2 * k] = valueReal * chirpReal + valueImag * chirpImag;
    spectrum[2 * k + 1] = valueImag * chirpReal - valueReal * chirpImag;
}

void launch_fft(
    const float* source,
    float* destination,
    int N,
    int threads,
    bool conjugateInput
) {
    int bits = 0;
    for (int value = N; value > 1; value >>= 1) ++bits;
    int blocks = (N + threads - 1) / threads;

    if (conjugateInput) {
        bit_reverse_conjugate_copy<<<blocks, threads>>>(
            source, destination, N, bits
        );
    } else {
        bit_reverse_copy<<<blocks, threads>>>(
            source, destination, N, bits
        );
    }

    int butterflies = N / 2;
    int butterflyBlocks =
        (butterflies + threads - 1) / threads;
    for (int length = 2; length <= N; length <<= 1) {
        fft_stage<<<butterflyBlocks, threads>>>(
            destination, N, length
        );
    }
}

// signal and spectrum are device pointers
extern "C" void solve(
    const float* signal,
    float* spectrum,
    int N
) {
    constexpr int threadsPerBlock = 256;
    bool powerOfTwo = N > 0 && (N & (N - 1)) == 0;

    if (powerOfTwo) {
        launch_fft(signal, spectrum, N, threadsPerBlock, false);
        cudaDeviceSynchronize();
        return;
    }

    int M = 1;
    while (M < 2 * N - 1) M <<= 1;
    size_t bytes = static_cast<size_t>(2) * M * sizeof(float);
    float* a = nullptr;
    float* b = nullptr;
    float* scratch = nullptr;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&scratch, bytes);
    cudaMemset(a, 0, bytes);
    cudaMemset(b, 0, bytes);

    int inputBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int workBlocks = (M + threadsPerBlock - 1) / threadsPerBlock;
    initialize_bluestein<<<inputBlocks, threadsPerBlock>>>(
        signal, a, b, N, M
    );

    launch_fft(a, scratch, M, threadsPerBlock, false);
    launch_fft(b, a, M, threadsPerBlock, false);
    multiply_complex<<<workBlocks, threadsPerBlock>>>(
        scratch, a, b, M
    );
    launch_fft(b, a, M, threadsPerBlock, true);
    finish_bluestein<<<inputBlocks, threadsPerBlock>>>(
        a, spectrum, N, M
    );

    cudaDeviceSynchronize();
    cudaFree(scratch);
    cudaFree(b);
    cudaFree(a);
}
```
