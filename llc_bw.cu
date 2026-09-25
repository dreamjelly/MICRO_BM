#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do {                                      \
  cudaError_t e = (x);                                          \
  if (e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(e));         \
    exit(1);                                                    \
  }                                                            \
} while (0)

extern "C" __global__
void init_kernel_u4(uint4* a, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < n; i += stride) {
    unsigned x = (unsigned)i;
    a[i] = make_uint4(x,
                      x * 1664525u + 1013904223u,
                      x ^ 0x5a5a5a5au,
                      ~x);
  }
}

extern "C" __global__
void llc_read_kernel_u4(const uint4* __restrict__ a,
                        uint4* __restrict__ sink,
                        size_t n,
                        int iters) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  uint4 acc = make_uint4(0, 0, 0, 0);

  for (int t = 0; t < iters; ++t) {
    // 每轮移动起点，避免编译器/局部缓存把访问过度简化
    size_t offset = ((size_t)t * 1024) % n;

    for (size_t i = tid; i < n; i += stride) {
      size_t j = i + offset;
      if (j >= n) j -= n;

      uint4 v = a[j];
      acc.x ^= v.x;
      acc.y ^= v.y;
      acc.z ^= v.z;
      acc.w ^= v.w;
    }
  }

  sink[tid] = acc;
}

int main(int argc, char** argv) {
  int mib = argc > 1 ? atoi(argv[1]) : 48;      // 默认 48 MiB，小于 64MB LLC
  int iters = argc > 2 ? atoi(argv[2]) : 5000;
  int blocks = argc > 3 ? atoi(argv[3]) : 1024;

  const int threads = 256;

  size_t bytes = (size_t)mib * 1024 * 1024;
  bytes = bytes / sizeof(uint4) * sizeof(uint4);
  size_t nvec = bytes / sizeof(uint4);

  uint4 *d_a = nullptr, *d_sink = nullptr;

  size_t sink_bytes = (size_t)blocks * threads * sizeof(uint4);

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, sink_bytes));

  init_kernel_u4<<<blocks, threads>>>(d_a, nvec);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup：把工作集尽量装入 LLC；不计入 acu profile range
  for (int w = 0; w < 5; ++w) {
    llc_read_kernel_u4<<<blocks, threads>>>(d_a, d_sink, nvec, 20);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaProfilerStart());

  CUDA_CHECK(cudaEventRecord(start));
  llc_read_kernel_u4<<<blocks, threads>>>(d_a, d_sink, nvec, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaProfilerStop());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  uint4 h;
  CUDA_CHECK(cudaMemcpy(&h, d_sink, sizeof(h), cudaMemcpyDeviceToHost));

  double user_bytes = (double)bytes * iters;
  double user_gbs = user_bytes / (ms / 1e3) / 1e9;

  printf("LLC read test\n");
  printf("working_set=%d MiB, iters=%d, blocks=%d, threads=%d\n",
         mib, iters, blocks, threads);
  printf("kernel_time=%.3f ms, requested_read_BW=%.3f GB/s\n",
         ms, user_gbs);
  printf("checksum=%u %u %u %u\n", h.x, h.y, h.z, h.w);

  cudaFree(d_a);
  cudaFree(d_sink);
  return 0;
}

// nvcc -O3 -arch=sm_80a llc_bw.cu -o llc_bw
// acu \
//   --profile-from-start off \
//   --metrics="regex:^llc.*$,regex:^l2__bytes.*$,regex:^kvd__bytes.*$,regex:^kvd__transactions.*$,regex:^dram.*$" \
//   --launch-count 1 \
//   --kill no \
//   -o llc_read \
//   ./llc_bw 48 5000 1024
