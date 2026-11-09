export const codeSamples = {
  "vector-addition-example-1": {
    fileName: "示例 1",
    label: "输入 / 输出",
    language: "text",
    code: `输入：A = [1.0, 2.0, 3.0, 4.0]
      B = [5.0, 6.0, 7.0, 8.0]
输出：C = [6.0, 8.0, 10.0, 12.0]`,
  },
  "vector-addition-example-2": {
    fileName: "示例 2",
    label: "输入 / 输出",
    language: "text",
    code: `输入：A = [1.5, 1.5, 1.5]
      B = [2.3, 2.3, 2.3]
输出：C = [3.8, 3.8, 3.8]`,
  },
  "vector-addition": {
    fileName: "vector_add.cu",
    label: "CUDA C++",
    language: "cpp",
    code: `#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}`,
  },
  "value-clipping": {
    fileName: "value_clipping.cu",
    label: "CUDA C++",
    language: "cpp",
    code: `__global__ void clip_kernel(
    const float* input,
    float* output,
    float lo,
    float hi,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x < lo ? lo : (x > hi ? hi : x);
    }
}`,
  },
  "matrix-addition": {
    fileName: "matrix_add.cu",
    label: "CUDA C++",
    language: "cpp",
    code: `__global__ void matrix_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (i < total) {
        C[i] = A[i] + B[i];
    }
}`,
  },
}
