#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <chrono>
using namespace std;

#define CUDA_CHECK(cmd) do {                          \
    cudaError_t e = cmd;                               \
    if( e != cudaSuccess ) {                            \
        printf("Failed: Cuda error %s:%d '%s'\n",        \
                __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                \
    }                                                       \
} while(0)

// KERNEL: Device-Initiated Send with device-side timing
__global__ void device_initiated_send(
    float4* __restrict__ dst,
    const float4* __restrict__ src,
    volatile int* ready_flag,
    size_t num_float4s,
    unsigned long long* elapsed_ns)
{
    unsigned long long start, end;
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    }
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_float4s; i += stride) {
        dst[i] = src[i];
    }
    
    __threadfence_system();
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *ready_flag = 1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
        *elapsed_ns = end - start;
    }
}

// KERNEL: Byte-level transfer with device-side timing
__global__ void device_initiated_send_bytes(
    char* __restrict__ dst,
    const char* __restrict__ src,
    volatile int* ready_flag,
    size_t num_bytes,
    unsigned long long* elapsed_ns)
{
    unsigned long long start, end;
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    }
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_bytes) {
        dst[idx] = src[idx];
    }
    
    __threadfence_system();
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *ready_flag = 1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
        *elapsed_ns = end - start;
    }
}

string format_size(size_t bytes) {
    if (bytes < 1024) return to_string(bytes) + "B";
    if (bytes < 1024*1024) return to_string(bytes/1024) + "KB";
    if (bytes < 1024*1024*1024) return to_string(bytes/(1024*1024)) + "MB";
    return to_string(bytes/(1024*1024*1024)) + "GB";
}

void benchmark_p2p_transfer(
    int src_gpu, 
    int dst_gpu, 
    size_t size_bytes,
    int iterations,
    int skip)
{
    CUDA_CHECK(cudaSetDevice(src_gpu));
    
    void *d_src;
    CUDA_CHECK(cudaMalloc(&d_src, size_bytes));
    CUDA_CHECK(cudaMemset(d_src, 0x42, size_bytes));
    
    
    CUDA_CHECK(cudaSetDevice(dst_gpu));
    
    void *d_dst;
    volatile int *d_ready_flag;
    unsigned long long *d_elapsed_ns;
    CUDA_CHECK(cudaMalloc(&d_dst, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_ready_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_elapsed_ns, sizeof(unsigned long long)));
    
    int blockSize, numBlocks;
    
    if (size_bytes < 16) {
        blockSize = 32;
        numBlocks = 1;
    } else if (size_bytes <= 1024) {
        blockSize = 32;
        size_t num_float4 = (size_bytes + 15) / 16;
        numBlocks = (num_float4 + blockSize - 1) / blockSize;
        numBlocks = (numBlocks > 4) ? 4 : numBlocks;
    } else if (size_bytes <= 64 * 1024) {
        blockSize = 256;
        size_t num_float4 = (size_bytes + 15) / 16;
        numBlocks = (num_float4 + blockSize - 1) / blockSize;
        numBlocks = (numBlocks > 32) ? 32 : numBlocks;
    } else {
        blockSize = 256;
        size_t num_float4 = (size_bytes + 15) / 16;
        numBlocks = (num_float4 + blockSize - 1) / blockSize;
        numBlocks = (numBlocks > 1024) ? 1024 : numBlocks;
    }
    
    CUDA_CHECK(cudaSetDevice(dst_gpu));
    CUDA_CHECK(cudaMemset((void*)d_ready_flag, 0, sizeof(int)));
    
    CUDA_CHECK(cudaSetDevice(src_gpu));
    
    for (int i = 0; i < skip; i++) {
        if (size_bytes < 16) {
            device_initiated_send_bytes<<<numBlocks, blockSize>>>(
                (char*)d_dst, (char*)d_src, d_ready_flag, size_bytes, d_elapsed_ns);
        } else {
            size_t num_float4 = (size_bytes + 15) / 16;
            device_initiated_send<<<numBlocks, blockSize>>>(
                (float4*)d_dst, (float4*)d_src, d_ready_flag, num_float4, d_elapsed_ns);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    double total_time_us = 0.0;
    
    for (int i = 0; i < (iterations - skip); i++) {
        if (size_bytes < 16) {
            device_initiated_send_bytes<<<numBlocks, blockSize>>>(
                (char*)d_dst, (char*)d_src, d_ready_flag, size_bytes, d_elapsed_ns);
        } else {
            size_t num_float4 = (size_bytes + 15) / 16;
            device_initiated_send<<<numBlocks, blockSize>>>(
                (float4*)d_dst, (float4*)d_src, d_ready_flag, num_float4, d_elapsed_ns);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long h_ns;
        CUDA_CHECK(cudaMemcpy(&h_ns, d_elapsed_ns, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        double time_us = h_ns / 1000.0;
        total_time_us += time_us;
    }
    
    double avg_time_us = total_time_us / (iterations - skip);
    double bandwidth_gbps = 0.0;
    if (avg_time_us > 0) {
        bandwidth_gbps = (size_bytes / (avg_time_us / 1e6)) / 1e9;
    }
    
    printf("%-15s\t%-15.3f\t%-15.2f\n", 
           format_size(size_bytes).c_str(),
           avg_time_us,
           bandwidth_gbps);
    
    CUDA_CHECK(cudaSetDevice(src_gpu));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaSetDevice(dst_gpu));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree((void*)d_ready_flag));
    CUDA_CHECK(cudaFree(d_elapsed_ns));
}

int main(int argc, char** argv) {
    printf("=============================================================\n");
    printf("GPU-Initiated P2P Transfer Benchmark with Doorbell Signaling\n");
    printf("=============================================================\n\n");
    
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count < 2) {
        fprintf(stderr, "Error: This benchmark requires at least 2 GPUs\n");
        fprintf(stderr, "Found: %d GPU(s)\n", device_count);
        return 1;
    }
    
    printf("Found %d GPU(s)\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    }
    
    int can_access_peer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, 0, 1));
    
    if (!can_access_peer) {
        fprintf(stderr, "\nError: GPU 0 cannot access GPU 1 via P2P\n");
        fprintf(stderr, "This benchmark requires P2P capability\n");
        return 1;
    }
    
    printf("\nP2P access: Enabled between GPU 0 and GPU 1\n");
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
    
    printf("\n=== P2P Bandwidth Test (GPU 0 -> GPU 1) ===\n");
    printf("%-15s\t%-15s\t%-15s\n", "Size", "Latency(μs)", "Bandwidth(GB/s)");
    printf("-------------------------------------------------------\n");
    
    const int ITERATIONS = 32;
    const int SKIP = 3;
    
    size_t min_size = 1;
    size_t max_size = 4ul * 1024 * 1024 * 1024;
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        benchmark_p2p_transfer(0, 1, size, ITERATIONS, SKIP);
    }
    
    return 0;
}