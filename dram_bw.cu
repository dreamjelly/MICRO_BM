#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CUDA_CHECK(x) do {                                      \
  cudaError_t e = (x);                                          \
  if (e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(e));         \
    exit(1);                                                    \
  }                                                            \
} while (0)

extern "C" __global__
void init_kernel_u4(uint4* a, uint4* b, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < n; i += stride) {
    unsigned x = (unsigned)i;
    uint4 v = make_uint4(x,
                         x * 1664525u + 1013904223u,
                         x ^ 0x5a5a5a5au,
                         ~x);
    if (a) a[i] = v;
    if (b) b[i] = make_uint4(0, 0, 0, 0);
  }
}

extern "C" __global__
void copy_kernel_u4(const uint4* __restrict__ a,
                    uint4* __restrict__ b,
                    size_t n,
                    int iters) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (int t = 0; t < iters; ++t) {
    for (size_t i = tid; i < n; i += stride) {
      uint4 v = a[i];
      b[i] = v;
    }
  }
}

extern "C" __global__
void read_kernel_u4(const uint4* __restrict__ a,
                    uint4* __restrict__ sink,
                    size_t n,
                    int iters) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  uint4 acc = make_uint4(0, 0, 0, 0);

  for (int t = 0; t < iters; ++t) {
    for (size_t i = tid; i < n; i += stride) {
      uint4 v = a[i];
      acc.x ^= v.x;
      acc.y ^= v.y;
      acc.z ^= v.z;
      acc.w ^= v.w;
    }
  }

  sink[tid] = acc;
}

extern "C" __global__
void write_kernel_u4(uint4* __restrict__ b,
                     size_t n,
                     int iters) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (int t = 0; t < iters; ++t) {
    for (size_t i = tid; i < n; i += stride) {
      unsigned x = (unsigned)(i + (size_t)t * n);
      b[i] = make_uint4(x, x + 1, x + 2, x + 3);
    }
  }
}

int main(int argc, char** argv) {
  double gib = argc > 1 ? atof(argv[1]) : 16.0;
  int iters = argc > 2 ? atoi(argv[2]) : 20;
  std::string mode = argc > 3 ? argv[3] : "copy";
  int blocks = argc > 4 ? atoi(argv[4]) : 1024;

  int threads = argc > 5 ? atoi(argv[5]) : 256;

  if (mode != "copy" && mode != "read" && mode != "write") {
    fprintf(stderr, "mode must be copy/read/write\n");
    return 1;
  }

  size_t bytes = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
  bytes = bytes / sizeof(uint4) * sizeof(uint4);
  size_t nvec = bytes / sizeof(uint4);

  bool need_a = mode != "write";
  bool need_b = mode != "read";

  size_t free_mem = 0, total_mem = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  size_t sink_bytes = (size_t)blocks * threads * sizeof(uint4);
  size_t need_bytes = sink_bytes;
  if (need_a) need_bytes += bytes;
  if (need_b) need_bytes += bytes;

  if (need_bytes > free_mem * 9 / 10) {
    fprintf(stderr,
            "Not enough memory. need %.2f GiB, free %.2f GiB\n",
            need_bytes / 1024.0 / 1024.0 / 1024.0,
            free_mem / 1024.0 / 1024.0 / 1024.0);
    return 1;
  }

  uint4 *d_a = nullptr, *d_b = nullptr, *d_sink = nullptr;

  if (need_a) CUDA_CHECK(cudaMalloc(&d_a, bytes));
  if (need_b) CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, sink_bytes));

  init_kernel_u4<<<blocks, threads>>>(d_a, d_b, nvec);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup，不放进 acu profile range
  if (mode == "copy") {
    copy_kernel_u4<<<blocks, threads>>>(d_a, d_b, nvec, 1);
  } else if (mode == "read") {
    read_kernel_u4<<<blocks, threads>>>(d_a, d_sink, nvec, 1);
  } else {
    write_kernel_u4<<<blocks, threads>>>(d_b, nvec, 1);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaProfilerStart());

  CUDA_CHECK(cudaEventRecord(start));
  if (mode == "copy") {
    copy_kernel_u4<<<blocks, threads>>>(d_a, d_b, nvec, iters);
  } else if (mode == "read") {
    read_kernel_u4<<<blocks, threads>>>(d_a, d_sink, nvec, iters);
  } else {
    write_kernel_u4<<<blocks, threads>>>(d_b, nvec, iters);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaProfilerStop());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double factor = mode == "copy" ? 2.0 : 1.0; // copy = read + write
  double user_bytes = (double)bytes * iters * factor;
  double user_gbs = user_bytes / (ms / 1e3) / 1e9;

  uint4 h;
  if (mode == "read") {
    CUDA_CHECK(cudaMemcpy(&h, d_sink, sizeof(h), cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpy(&h, d_b + nvec / 2, sizeof(h), cudaMemcpyDeviceToHost));
  }

  printf("mode=%s, array=%.2f GiB, iters=%d, blocks=%d, threads=%d\n",
         mode.c_str(), bytes / 1024.0 / 1024.0 / 1024.0,
         iters, blocks, threads);
  printf("kernel_time=%.3f ms, requested_user_BW=%.3f GB/s\n",
         ms, user_gbs);
  printf("checksum=%u %u %u %u\n", h.x, h.y, h.z, h.w);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_sink);
  return 0;
}
