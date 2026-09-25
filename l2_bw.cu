// l2_bw_private.cu
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

#define CHECK_CUDA(x) do {                                      \
  cudaError_t err = (x);                                        \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));       \
    exit(1);                                                    \
  }                                                            \
} while (0)

static int get_int_arg(int argc, char** argv, const char* key, int defv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
  }
  return defv;
}

extern "C" __global__
void init_u4(uint4* data, size_t nvec) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < nvec; i += stride) {
    unsigned x = (unsigned)i;
    data[i] = make_uint4(x, x * 1664525u + 1013904223u,
                         x ^ 0x5a5a5a5au, ~x);
  }
}

extern "C" __global__
void l2_bw_private_kernel(const uint4* __restrict__ data,
                          uint4* __restrict__ sink,
                          size_t vec_per_block,
                          int iters) {
  extern __shared__ unsigned char smem[];

  if (threadIdx.x == 0) {
    smem[0] = (unsigned char)blockIdx.x;
  }

  size_t base = (size_t)blockIdx.x * vec_per_block;
  uint4 acc = make_uint4(0, 0, 0, 0);

  for (int r = 0; r < iters; ++r) {
    size_t offset = ((size_t)r * 257) % vec_per_block;

    for (size_t i = threadIdx.x; i < vec_per_block; i += blockDim.x) {
      size_t j = i + offset;
      if (j >= vec_per_block) j -= vec_per_block;

      uint4 v = data[base + j];

      acc.x ^= v.x;
      acc.y ^= v.y;
      acc.z ^= v.z;
      acc.w ^= v.w;
    }
  }

  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main(int argc, char** argv) {
  int blocks = get_int_arg(argc, argv, "--blocks", 64);
  int threads = get_int_arg(argc, argv, "--threads", 512);
  int workset_kb = get_int_arg(argc, argv, "--workset-kb", 192);
  int iters = get_int_arg(argc, argv, "--iters", 5000);
  int smem_kb = get_int_arg(argc, argv, "--smem-kb", 160);

  size_t bytes_per_block = (size_t)workset_kb * 1024;
  bytes_per_block = bytes_per_block / sizeof(uint4) * sizeof(uint4);

  size_t vec_per_block = bytes_per_block / sizeof(uint4);
  size_t total_vec = (size_t)blocks * vec_per_block;

  uint4* d_data = nullptr;
  uint4* d_sink = nullptr;

  CHECK_CUDA(cudaMalloc(&d_data, total_vec * sizeof(uint4)));
  CHECK_CUDA(cudaMalloc(&d_sink, (size_t)blocks * threads * sizeof(uint4)));

  init_u4<<<blocks, threads>>>(d_data, total_vec);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  size_t smem_bytes = (size_t)smem_kb * 1024;

  CHECK_CUDA(cudaFuncSetAttribute(
      l2_bw_private_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes));

  // warmup：装入 L2/LLC，不计入 acu profile range
  for (int w = 0; w < 5; ++w) {
    l2_bw_private_kernel<<<blocks, threads, smem_bytes>>>(
        d_data, d_sink, vec_per_block, 200);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t st, ed;
  CHECK_CUDA(cudaEventCreate(&st));
  CHECK_CUDA(cudaEventCreate(&ed));

  CHECK_CUDA(cudaProfilerStart());

  CHECK_CUDA(cudaEventRecord(st));
  l2_bw_private_kernel<<<blocks, threads, smem_bytes>>>(
      d_data, d_sink, vec_per_block, iters);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(ed));
  CHECK_CUDA(cudaEventSynchronize(ed));

  CHECK_CUDA(cudaProfilerStop());

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));

  uint4 h;
  CHECK_CUDA(cudaMemcpy(&h, d_sink, sizeof(uint4), cudaMemcpyDeviceToHost));

  double user_bytes = (double)bytes_per_block * blocks * iters;
  double user_gbs = user_bytes / (ms / 1e3) / 1e9;

  printf("L2 private read BW test\n");
  printf("blocks=%d threads=%d workset=%dKB/block iters=%d smem=%dKB\n",
         blocks, threads, workset_kb, iters, smem_kb);
  printf("total_workset=%.2f MiB, per_L2_cluster_est=%.2f KiB\n",
         (double)bytes_per_block * blocks / 1024.0 / 1024.0,
         (double)bytes_per_block * 4 / 1024.0);
  printf("kernel_time=%.3f ms, requested_read_BW=%.3f GB/s\n",
         ms, user_gbs);
  printf("checksum=%u %u %u %u\n", h.x, h.y, h.z, h.w);

  cudaFree(d_data);
  cudaFree(d_sink);
  return 0;
}

// nvcc -O3 -arch=sm_80a l2_bw.cu -o l2_bw
// acu -f \
//   --profile-from-start off \
//   --metrics="regex:^l2.*$,regex:^kvd__bytes.*$,regex:^kvd__transactions.*$,regex:^llc.*$,regex:^dram.*$,regex:^launch.*$" \
//   --launch-count 1 \
//   --kill no \
//   --page raw \
//   --csv-file l2_bw_192k.csv \
//   -o l2_bw_192k \
//   ./l2_bw --blocks 64 --threads 512 --workset-kb 192 --iters 5000 --smem-kb 160
