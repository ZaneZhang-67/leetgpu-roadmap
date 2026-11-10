export const codeSamples = {
  "vector-addition-example-1": {
    fileName: "示例 1",
    label: "输入 / 输出",
    language: "text",
    plain: true,
    code: `输入：A = [1.0, 2.0, 3.0, 4.0]
      B = [5.0, 6.0, 7.0, 8.0]
输出：C = [6.0, 8.0, 10.0, 12.0]`,
  },
  "vector-addition-example-2": {
    fileName: "示例 2",
    label: "输入 / 输出",
    language: "text",
    plain: true,
    code: `输入：A = [1.5, 1.5, 1.5]
      B = [2.3, 2.3, 2.3]
输出：C = [3.8, 3.8, 3.8]`,
  },
  "vector-addition": {
    fileName: "vector_add.cu",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
    code: `#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int nx = bx * blockDim.x + tx;

    if (nx < N) {
        C[nx] = A[nx] + B[nx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}`,
  },
  "vector-addition-index": {
    fileName: "全局下标",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
    code: `int i = blockIdx.x * blockDim.x + threadIdx.x;`,
  },
  "vector-addition-grid-size": {
    fileName: "Grid 大小",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
    code: `int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;`,
  },
  "vector-addition-boundary": {
    fileName: "边界检查",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
    code: `if (i < N)`,
  },
  "vector-addition-operation": {
    fileName: "逐元素加法",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
    code: `C[i] = A[i] + B[i];`,
  },
  "vector-addition-mapping": {
    fileName: "线程映射",
    label: "示意",
    language: "text",
    plain: true,
    code: `线程 0  → C[0] = A[0] + B[0]
线程 1  → C[1] = A[1] + B[1]
线程 2  → C[2] = A[2] + B[2]
...
线程 i  → C[i] = A[i] + B[i]`,
  },
  "vector-addition-cpu": {
    fileName: "CPU 写法",
    label: "C++",
    language: "cpp",
    lineNumbers: true,
    code: `for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
}`,
  },
  "vector-addition-gpu": {
    fileName: "GPU 写法",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
    code: `int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
    C[i] = A[i] + B[i];
}`,
  },
  "value-clipping": {
    fileName: "value_clipping.cu",
    label: "CUDA C++",
    language: "cpp",
    lineNumbers: true,
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
    lineNumbers: true,
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
