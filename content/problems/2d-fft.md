---
id: 2d-fft
leetgpuId: 78
titleEn: 2D FFT
titleZh: 二维快速傅里叶变换
description: 使用 CUDA 对实部与虚部交错存储的复数矩阵执行二维快速傅里叶变换，覆盖任意允许尺寸
problemUrl: https://leetgpu.com/challenges/2d-fft
updatedAt: 2026-11-12
---
## 题目定义

给定形状为 `M × N` 的复数信号，计算未归一化的二维正向离散傅里叶变换：

```text
X[k,l] = Σ_m Σ_n x[m,n] × exp(-2πi(km/M + ln/N))
```

输入 `signal` 和输出 `spectrum` 都是长度为 `2 × M × N` 的 FP32 一维数组。复数采用行主序交错存储：`x[m,n]` 的实部位于 `2 × (m × N + n)`，虚部位于后一位置。

### 实现要求

- 只使用原生功能，不允许使用外部库
- `solve` 函数签名必须保持不变
- 结果写入 `spectrum`
- 同时变换行维和列维，不执行逆变换缩放

### 示例

```text
M = 2，N = 2
输入实部 = [[1, 0], [0, 0]]
输入虚部 = [[0, 0], [0, 0]]
输出实部 = [[1, 1], [1, 1]]
输出虚部 = [[0, 0], [0, 0]]
```

### 约束

- `1 ≤ M, N ≤ 4096`
- 实部和虚部均为 FP32
- 功能测试包含 `1×1`、`1×4`、`3×5`、`30×30`、`100×200`、`512×512`
- 性能测试使用 `M = N = 2048`
- 绝对误差和相对误差均为 `0.01`

## 模板代码解析

CUDA starter 只提供固定入口：

```cpp
#include <cuda_runtime.h>

// signal, spectrum are device pointers
extern "C" void solve(
    const float* signal,
    float* spectrum,
    int M,
    int N
) {}
```

`signal` 和 `spectrum` 已在设备端。starter 没有提供 kernel、工作区或启动配置，作答需要自行完成两维变换和中间结果管理。

## 实现思路

二维 FFT 可以按行列分解。先对 `M` 行分别执行长度 `N` 的正向 FFT，再将结果转置为 `N × M`；随后对转置后的 `N` 行分别执行长度 `M` 的正向 FFT，最后再次转置，写回原布局。转置后，原本跨步存储的列变成连续行，无需为列 FFT 编写低效的 stride 访问。

每个一维变换都有两条路径：长度为 2 的幂时执行 radix-2 FFT，其他长度则用 Bluestein 恒等式归约为长度 `P = nextPow2(2L-1)` 的循环卷积。工作数组使用 FP64，输入和最终输出仍为 FP32。正向 FFT 使用负相位，不除以长度；只有 Bluestein 内部的逆 FFT 除以 `P`。

批量 radix-2 的每个阶段都由一个 kernel 覆盖所有行和蝶形，kernel 启动边界负责阶段间的全局同步。对于长度相同的各行，Bluestein 复用同一份 chirp 频谱，并为每行准备卷积工作区。最大测试 `2048×2048` 的两个维度均走 radix-2，计算量为 `O(MN(log M+log N))`，省去了直接 DFT 所需的约 `2×2048³` 个复数项。

## 参考实现

```cpp
#include <cuda_runtime.h>
#include <math.h>

__device__ unsigned reverse_bits(unsigned value, int bits) {
    unsigned result = 0;
    for (int i = 0; i < bits; ++i) {
        result = (result << 1) | (value & 1u);
        value >>= 1;
    }
    return result;
}

__global__ void load_bit_reversed(
    const float* source, double2* data,
    int rows, int length, int bits
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * length;
    if (index >= total) return;
    int row = index / length;
    int column = index % length;
    int destination = row * length + reverse_bits(column, bits);
    data[destination] = make_double2(
        source[2 * index], source[2 * index + 1]
    );
}

__global__ void copy_bit_reversed(
    const double2* source, double2* destination,
    int rows, int length, int bits, bool conjugate
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * length;
    if (index >= total) return;
    int row = index / length;
    int column = index % length;
    int target = row * length + reverse_bits(column, bits);
    double2 value = source[index];
    if (conjugate) value.y = -value.y;
    destination[target] = value;
}

__global__ void fft_stage(
    double2* data, int rows, int length, int span
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int butterfliesPerRow = length / 2;
    int total = rows * butterfliesPerRow;
    if (index >= total) return;
    int row = index / butterfliesPerRow;
    int butterfly = index % butterfliesPerRow;
    int half = span / 2;
    int group = butterfly / half;
    int offset = butterfly % half;
    int left = row * length + group * span + offset;
    int right = left + half;
    double sine, cosine;
    sincos(-6.28318530717958647692 * offset / span, &sine, &cosine);
    double2 a = data[left];
    double2 b = data[right];
    double2 t = make_double2(
        cosine * b.x - sine * b.y,
        sine * b.x + cosine * b.y
    );
    data[left] = make_double2(a.x + t.x, a.y + t.y);
    data[right] = make_double2(a.x - t.x, a.y - t.y);
}

void run_fft_stages(double2* data, int rows, int length) {
    constexpr int threads = 256;
    int total = rows * (length / 2);
    int blocks = (total + threads - 1) / threads;
    for (int span = 2; span <= length; span <<= 1) {
        fft_stage<<<blocks, threads>>>(data, rows, length, span);
    }
}

__global__ void store_float(
    const double2* source, float* destination, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[2 * index] = static_cast<float>(source[index].x);
    destination[2 * index + 1] = static_cast<float>(source[index].y);
}

__global__ void initialize_chirp(
    double2* chirp, int length, int padded
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= length) return;
    unsigned long long square =
        static_cast<unsigned long long>(n) * n;
    unsigned long long phase = square % (2ull * length);
    double sine, cosine;
    sincos(3.14159265358979323846 * phase / length, &sine, &cosine);
    chirp[n] = make_double2(cosine, sine);
    if (n > 0) chirp[padded - n] = make_double2(cosine, sine);
}

__global__ void initialize_bluestein_rows(
    const float* source, double2* values,
    int rows, int length, int padded
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * length;
    if (index >= total) return;
    int row = index / length;
    int n = index % length;
    unsigned long long square =
        static_cast<unsigned long long>(n) * n;
    unsigned long long phase = square % (2ull * length);
    double sine, cosine;
    sincos(3.14159265358979323846 * phase / length, &sine, &cosine);
    double real = source[2 * index];
    double imaginary = source[2 * index + 1];
    values[row * padded + n] = make_double2(
        real * cosine + imaginary * sine,
        imaginary * cosine - real * sine
    );
}

__global__ void multiply_rows_by_chirp(
    double2* values, const double2* chirp,
    int rows, int padded
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * padded;
    if (index >= total) return;
    double2 a = values[index];
    double2 b = chirp[index % padded];
    values[index] = make_double2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__global__ void finish_bluestein_rows(
    const double2* values, float* destination,
    int rows, int length, int padded
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * length;
    if (index >= total) return;
    int row = index / length;
    int k = index % length;
    unsigned long long square =
        static_cast<unsigned long long>(k) * k;
    unsigned long long phase = square % (2ull * length);
    double sine, cosine;
    sincos(3.14159265358979323846 * phase / length, &sine, &cosine);
    double2 value = values[row * padded + k];
    double real = value.x / padded;
    double imaginary = -value.y / padded;
    destination[2 * index] = static_cast<float>(
        real * cosine + imaginary * sine
    );
    destination[2 * index + 1] = static_cast<float>(
        imaginary * cosine - real * sine
    );
}

void batch_fft(const float* source, float* destination, int rows, int length) {
    constexpr int threads = 256;
    int count = rows * length;
    bool powerOfTwo = (length & (length - 1)) == 0;
    if (powerOfTwo) {
        int bits = 0;
        for (int value = length; value > 1; value >>= 1) ++bits;
        double2* work = nullptr;
        cudaMalloc(&work, static_cast<size_t>(count) * sizeof(double2));
        int blocks = (count + threads - 1) / threads;
        load_bit_reversed<<<blocks, threads>>>(
            source, work, rows, length, bits
        );
        run_fft_stages(work, rows, length);
        store_float<<<blocks, threads>>>(work, destination, count);
        cudaFree(work);
        return;
    }

    int padded = 1;
    while (padded < 2 * length - 1) padded <<= 1;
    size_t rowBytes = static_cast<size_t>(rows) * padded * sizeof(double2);
    size_t chirpBytes = static_cast<size_t>(padded) * sizeof(double2);
    double2 *values = nullptr, *scratch = nullptr, *chirp = nullptr, *chirpFft = nullptr;
    cudaMalloc(&values, rowBytes);
    cudaMalloc(&scratch, rowBytes);
    cudaMalloc(&chirp, chirpBytes);
    cudaMalloc(&chirpFft, chirpBytes);
    cudaMemset(values, 0, rowBytes);
    cudaMemset(chirp, 0, chirpBytes);
    int inputBlocks = (count + threads - 1) / threads;
    int workBlocks = (rows * padded + threads - 1) / threads;
    int chirpBlocks = (padded + threads - 1) / threads;
    initialize_bluestein_rows<<<inputBlocks, threads>>>(
        source, values, rows, length, padded
    );
    initialize_chirp<<<
        (length + threads - 1) / threads, threads
    >>>(chirp, length, padded);

    int bits = 0;
    for (int value = padded; value > 1; value >>= 1) ++bits;
    copy_bit_reversed<<<workBlocks, threads>>>(
        values, scratch, rows, padded, bits, false
    );
    run_fft_stages(scratch, rows, padded);
    copy_bit_reversed<<<chirpBlocks, threads>>>(
        chirp, chirpFft, 1, padded, bits, false
    );
    run_fft_stages(chirpFft, 1, padded);
    multiply_rows_by_chirp<<<workBlocks, threads>>>(
        scratch, chirpFft, rows, padded
    );
    copy_bit_reversed<<<workBlocks, threads>>>(
        scratch, values, rows, padded, bits, true
    );
    run_fft_stages(values, rows, padded);
    finish_bluestein_rows<<<inputBlocks, threads>>>(
        values, destination, rows, length, padded
    );
    cudaFree(chirpFft);
    cudaFree(chirp);
    cudaFree(scratch);
    cudaFree(values);
}

__global__ void transpose_complex(
    const float* source, float* destination, int rows, int columns
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * columns;
    if (index >= total) return;
    int row = index / columns;
    int column = index % columns;
    int target = column * rows + row;
    destination[2 * target] = source[2 * index];
    destination[2 * target + 1] = source[2 * index + 1];
}

// signal, spectrum are device pointers
extern "C" void solve(
    const float* signal,
    float* spectrum,
    int M,
    int N
) {
    size_t bytes = static_cast<size_t>(2) * M * N * sizeof(float);
    float *rowSpectrum = nullptr, *transposed = nullptr, *columnSpectrum = nullptr;
    cudaMalloc(&rowSpectrum, bytes);
    cudaMalloc(&transposed, bytes);
    cudaMalloc(&columnSpectrum, bytes);

    batch_fft(signal, rowSpectrum, M, N);
    constexpr int threads = 256;
    int total = M * N;
    int blocks = (total + threads - 1) / threads;
    transpose_complex<<<blocks, threads>>>(rowSpectrum, transposed, M, N);
    batch_fft(transposed, columnSpectrum, N, M);
    transpose_complex<<<blocks, threads>>>(columnSpectrum, spectrum, N, M);

    cudaDeviceSynchronize();
    cudaFree(columnSpectrum);
    cudaFree(transposed);
    cudaFree(rowSpectrum);
}
```
