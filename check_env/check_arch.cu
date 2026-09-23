#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("SM arch: sm_%d%d\n", prop.major, prop.minor);
        printf("SM count: %d\n", prop.multiProcessorCount);
    }
}
// nvcc check_arch.cu -o check_arch
// ./check_arch