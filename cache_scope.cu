// cache_scope.cu
// nvcc -O3 -lineinfo cache_scope.cu -o cache_scope

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CHECK_CUDA(x) do {                                      \
    cudaError_t err = (x);                                      \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(1);                                                \
    }                                                           \
} while (0)

static int get_int_arg(int argc, char** argv, const char* key, int defv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    }
    return defv;
}

static const char* get_str_arg(int argc, char** argv, const char* key, const char* defv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return defv;
}

__global__ void flush_kernel(const uint32_t* __restrict__ in,
                             uint64_t n_words,
                             unsigned long long* out) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    unsigned long long sum = 0;

    for (uint64_t i = tid; i < n_words; i += stride) {
        sum += in[i];
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out, sum);
    }
}

// mode = 0: shared, all blocks read same working set.
// mode = 1: private, each block reads its own working set.
extern "C" __global__
void cache_scope_kernel(const uint32_t* __restrict__ data,
                        uint64_t words_per_set,
                        int mode,
                        int iters,
                        int stride_words,
                        unsigned long long* out) {
    extern __shared__ unsigned char smem[];

    // 只用于占用 shared/TSM，帮助降低每 CU 并发 block 数。
    if (threadIdx.x == 0 && blockDim.x > 0) {
        smem[0] = (unsigned char)blockIdx.x;
    }

    uint64_t base = (mode == 0) ? 0 : (uint64_t)blockIdx.x * words_per_set;
    unsigned long long sum = 0;

    // 每个 thread 以 cacheline stride 访问，避免一次顺序流式带宽掩盖 cache 行为。
    for (int r = 0; r < iters; ++r) {
        for (uint64_t i = (uint64_t)threadIdx.x * stride_words;
             i < words_per_set;
             i += (uint64_t)blockDim.x * stride_words) {
            sum += data[base + i];
        }
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&out[blockIdx.x], sum);
    }
}

int main(int argc, char** argv) {
    int blocks      = get_int_arg(argc, argv, "--blocks", 1);
    int threads     = get_int_arg(argc, argv, "--threads", 256);
    int workset_kb  = get_int_arg(argc, argv, "--workset-kb", 512);
    int iters       = get_int_arg(argc, argv, "--iters", 2000);
    int stride_b    = get_int_arg(argc, argv, "--stride-bytes", 64);
    int flush_mb    = get_int_arg(argc, argv, "--flush-mb", 256);
    int smem_kb     = get_int_arg(argc, argv, "--smem-kb", 160);
    const char* mode_s = get_str_arg(argc, argv, "--mode", "shared");

    int mode = 0;
    if (strcmp(mode_s, "private") == 0) mode = 1;
    else if (strcmp(mode_s, "shared") == 0) mode = 0;
    else {
        fprintf(stderr, "mode must be shared/private\n");
        return 1;
    }

    uint64_t words_per_set = ((uint64_t)workset_kb * 1024) / sizeof(uint32_t);
    int stride_words = stride_b / (int)sizeof(uint32_t);
    if (stride_words < 1) stride_words = 1;

    uint64_t total_sets = (mode == 0) ? 1 : blocks;
    uint64_t total_words = total_sets * words_per_set;

    uint32_t* d_data = nullptr;
    unsigned long long* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, total_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, blocks * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_data, 1, total_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_out, 0, blocks * sizeof(unsigned long long)));

    uint32_t* d_flush = nullptr;
    unsigned long long* d_flush_out = nullptr;

    if (flush_mb > 0) {
        uint64_t flush_words = ((uint64_t)flush_mb * 1024 * 1024) / sizeof(uint32_t);
        CHECK_CUDA(cudaMalloc(&d_flush, flush_words * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_flush_out, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_flush, 7, flush_words * sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(d_flush_out, 0, sizeof(unsigned long long)));

        int flush_blocks = 4096;
        flush_kernel<<<flush_blocks, 256>>>(d_flush, flush_words, d_flush_out);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    size_t smem_bytes = (size_t)smem_kb * 1024;

    // 如果 smem_kb 较大，需要开启动态 shared 上限。
    cudaFuncSetAttribute(
        cache_scope_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes
    );

    cudaEvent_t st, ed;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&ed));

    CHECK_CUDA(cudaEventRecord(st));
    cache_scope_kernel<<<blocks, threads, smem_bytes>>>(
        d_data, words_per_set, mode, iters, stride_words, d_out
    );
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));

    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));

    unsigned long long* h_out =
        (unsigned long long*)malloc(blocks * sizeof(unsigned long long));
    CHECK_CUDA(cudaMemcpy(h_out, d_out,
                          blocks * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    unsigned long long total = 0;
    for (int i = 0; i < blocks; ++i) total += h_out[i];

    printf("mode=%s blocks=%d workset_kb=%d iters=%d stride=%dB smem=%dKB time_ms=%.3f checksum=%llu\n",
           mode_s, blocks, workset_kb, iters, stride_b, smem_kb, ms, total);

    free(h_out);
    cudaFree(d_data);
    cudaFree(d_out);
    if (d_flush) cudaFree(d_flush);
    if (d_flush_out) cudaFree(d_flush_out);

    return 0;
}
