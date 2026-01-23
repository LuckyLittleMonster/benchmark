/*
 * Small Message GPU Bandwidth Test - Cache Bypass Version
 * 
 * This program tests bandwidth for small messages with L2 cache bypass.
 * Each iteration accesses different data chunks to avoid cache hits.
 * Uses clock64() for high-precision timing inside kernels.
 * 
 * Comparison scenarios:
 * 1. GPU 0 -> GPU 0 (Device-to-Device internal, baseline) [Always runs]
 * 2. CPU Memory -> GPU 0 (GPU accessing CPU via UMA read) [Always runs]
 * 3. GPU 0 -> CPU Memory (GPU accessing CPU via UMA write) [Always runs]
 * 4. GPU 0 -> GPU 1 (GPU Direct P2P) [Requires 2+ GPUs with P2P support]
 * 5. CPU Memory -> GPU 1 (GPU accessing CPU via UMA read) [Requires 2+ GPUs with P2P support]
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>

using std::cout;
using std::endl;
using std::string;

string ljust(string s, int width) {
    return s + string(width - s.size(), ' ');
}

#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",     \
                __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)

// Kernel with cache bypass - accesses different chunks each iteration
__global__ void memcpy_kernel_nocache(float* __restrict__ dst, 
                                       const float* __restrict__ src, 
                                       size_t chunk_size,
                                       int num_chunks,
                                       unsigned long long* cycles_per_thread,
                                       int skip_iterations,
                                       int timed_iterations) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    size_t floats_per_thread = (chunk_size + block_size - 1) / block_size;
    size_t start_idx = tid * floats_per_thread;
    size_t end_idx = min(start_idx + floats_per_thread, chunk_size);
    
    // Warm-up iterations
    for (int iter = 0; iter < skip_iterations; iter++) {
        size_t src_offset = (iter % num_chunks) * chunk_size;
        size_t dst_offset = (iter % num_chunks) * chunk_size;
        for (size_t i = start_idx; i < end_idx; i++) {
            dst[dst_offset + i] = src[src_offset + i];
        }
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations - access different chunks
    for (int iter = 0; iter < timed_iterations; iter++) {
        size_t src_offset = ((skip_iterations + iter) % num_chunks) * chunk_size;
        size_t dst_offset = ((skip_iterations + iter) % num_chunks) * chunk_size;
        for (size_t i = start_idx; i < end_idx; i++) {
            dst[dst_offset + i] = src[src_offset + i];
        }
    }
    
    __syncthreads();
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_per_thread[0] = end_cycle - start_cycle;
    }
}

// Vectorized version using float4
__global__ void memcpy_kernel_nocache_float4(float* __restrict__ dst, 
                                              const float* __restrict__ src, 
                                              size_t chunk_size,
                                              int num_chunks,
                                              unsigned long long* cycles_per_thread,
                                              int skip_iterations,
                                              int timed_iterations) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    size_t num_float4 = chunk_size / 4;
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    
    size_t float4_per_thread = (num_float4 + block_size - 1) / block_size;
    size_t start_idx = tid * float4_per_thread;
    size_t end_idx = min(start_idx + float4_per_thread, num_float4);
    
    // Warm-up iterations
    for (int iter = 0; iter < skip_iterations; iter++) {
        size_t chunk_offset = (iter % num_chunks) * num_float4;
        for (size_t i = start_idx; i < end_idx; i++) {
            dst4[chunk_offset + i] = src4[chunk_offset + i];
        }
        
        if (tid == 0) {
            size_t base = (iter % num_chunks) * chunk_size;
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < chunk_size; i++) {
                dst[base + i] = src[base + i];
            }
        }
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        size_t chunk_offset = ((skip_iterations + iter) % num_chunks) * num_float4;
        for (size_t i = start_idx; i < end_idx; i++) {
            dst4[chunk_offset + i] = src4[chunk_offset + i];
        }
        
        if (tid == 0) {
            size_t base = ((skip_iterations + iter) % num_chunks) * chunk_size;
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < chunk_size; i++) {
                dst[base + i] = src[base + i];
            }
        }
    }
    
    __syncthreads();
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_per_thread[0] = end_cycle - start_cycle;
    }
}

void test_nocache_bandwidth(const string& test_name,
                             float* src, float* dst,
                             size_t min_chunk_size, size_t max_chunk_size,
                             size_t total_buffer_size,
                             int iterations, int skip,
                             double clock_rate_ghz,
                             int device_id) {
    
    cout << "\n=== " << test_name << " ===" << endl;
    cout << ljust("size(B)", 15) << "\t" 
         << ljust("cycles", 18) << "\t"
         << ljust("avg_lat(ns)", 15) << "\t" 
         << ljust("total_time(us)", 18) << "\t"
         << ljust("bw(GB/s)", 15) << endl;
    
    unsigned long long* d_cycles;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    
    unsigned long long h_cycles;
    int timed_iterations = iterations - skip;
    
    for (size_t chunk_size = min_chunk_size; chunk_size <= max_chunk_size; chunk_size *= 2) {
        // Calculate number of chunks
        int num_chunks = total_buffer_size / chunk_size;
        if (num_chunks < 10) num_chunks = 10;
        
        size_t num_floats = chunk_size / sizeof(float);
        
        int block_size = 256;
        if (num_floats < 256) {
            block_size = 32;
        }
        
        // Single kernel call with cache bypass
        if (chunk_size >= 16) {
            memcpy_kernel_nocache_float4<<<1, block_size>>>(
                dst, src, num_floats, num_chunks, d_cycles, skip, timed_iterations);
        } else {
            memcpy_kernel_nocache<<<1, block_size>>>(
                dst, src, num_floats, num_chunks, d_cycles, skip, timed_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), 
                              cudaMemcpyDeviceToHost));
        
        // Calculate metrics
        double total_time_ns = h_cycles / clock_rate_ghz;
        double total_time_us = total_time_ns / 1000.0;
        double avg_latency_ns = total_time_ns / timed_iterations;
        double total_time_s = total_time_ns / 1e9;
        double bandwidth = (chunk_size * timed_iterations / (1024.0 * 1024.0 * 1024.0)) / total_time_s;
        
        cout << ljust(std::to_string(chunk_size), 15) << "\t"
             << ljust(std::to_string(h_cycles), 18) << "\t"
             << ljust(std::to_string(avg_latency_ns), 15) << "\t"
             << ljust(std::to_string(total_time_us), 18) << "\t"
             << ljust(std::to_string(bandwidth), 15) << endl;
    }
    
    CUDA_CHECK(cudaFree(d_cycles));
}

int main(int argc, char *argv[]) {
    // Test parameters
    size_t min_chunk_size = 4;           // 4B
    size_t max_chunk_size = 64 * 1024;   // 64KB
    size_t total_buffer_size = 128 * 1024 * 1024;  // 128MB total buffer (>> L2 cache)
    int iterations = 1000;
    int skip = 10;
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    cout << "Found " << deviceCount << " GPU(s)" << endl;
    
    if (deviceCount < 1) {
        cout << "Error: Need at least 1 GPU for this test" << endl;
        return 1;
    }
    
    cudaDeviceProp prop0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
    double clock_rate_ghz_0 = prop0.clockRate / 1e6;
    
    cout << "\nGPU 0: " << prop0.name << endl;
    cout << "  Clock rate: " << clock_rate_ghz_0 << " GHz" << endl;
    cout << "  L2 Cache: " << prop0.l2CacheSize / 1024 / 1024 << " MB" << endl;
    cout << "  Unified Addressing: " << (prop0.unifiedAddressing ? "Yes" : "No") << endl;
    
    double clock_rate_ghz_1 = 0.0;
    bool canAccessPeer = false;
    
    if (deviceCount >= 2) {
        cudaDeviceProp prop1;
        CUDA_CHECK(cudaGetDeviceProperties(&prop1, 1));
        clock_rate_ghz_1 = prop1.clockRate / 1e6;
        
        cout << "\nGPU 1: " << prop1.name << endl;
        cout << "  Clock rate: " << clock_rate_ghz_1 << " GHz" << endl;
        
        int canAccessPeerInt;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeerInt, 0, 1));
        canAccessPeer = (canAccessPeerInt != 0);
        cout << "\nP2P Access (GPU 0 -> GPU 1): " << (canAccessPeer ? "Supported" : "Not supported") << endl;
    } else {
        cout << "\nOnly 1 GPU found. Will skip multi-GPU tests." << endl;
    }
    
    // Allocate large buffers on GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    float *gpu0_src, *gpu0_dst;
    CUDA_CHECK(cudaMalloc(&gpu0_src, total_buffer_size));
    CUDA_CHECK(cudaMalloc(&gpu0_dst, total_buffer_size));
    CUDA_CHECK(cudaMemset(gpu0_src, 0, total_buffer_size));
    CUDA_CHECK(cudaMemset(gpu0_dst, 0, total_buffer_size));
    
    // Allocate large buffers on GPU 1 (if available)
    float *gpu1_dst = nullptr;
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMalloc(&gpu1_dst, total_buffer_size));
        CUDA_CHECK(cudaMemset(gpu1_dst, 0, total_buffer_size));
    }
    
    // Allocate large CPU pinned memory
    float *cpu_buf;
    CUDA_CHECK(cudaHostAlloc(&cpu_buf, total_buffer_size, cudaHostAllocMapped));
    memset(cpu_buf, 0, total_buffer_size);
    
    float *cpu_buf_dev0, *cpu_buf_dev1 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaHostGetDevicePointer(&cpu_buf_dev0, cpu_buf, 0));
    
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaHostGetDevicePointer(&cpu_buf_dev1, cpu_buf, 0));
    }
    
    // Enable P2P if available
    if (deviceCount >= 2 && canAccessPeer) {
        CUDA_CHECK(cudaSetDevice(0));
        cudaError_t peer_err = cudaDeviceEnablePeerAccess(1, 0);
        if (peer_err == cudaSuccess) {
            cout << "\nP2P enabled: GPU 0 -> GPU 1\n" << endl;
        } else if (peer_err == cudaErrorPeerAccessAlreadyEnabled) {
            cout << "\nP2P already enabled: GPU 0 -> GPU 1\n" << endl;
            cudaGetLastError();
        } else {
            CUDA_CHECK(peer_err);
        }
    }
    
    // Test 1: GPU 0 -> GPU 0 (baseline)
    CUDA_CHECK(cudaSetDevice(0));
    test_nocache_bandwidth(
        "Test 1: GPU 0 -> GPU 0 (D2D internal, baseline)",
        gpu0_src, gpu0_dst,
        min_chunk_size, max_chunk_size, total_buffer_size,
        iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Test 2: CPU Memory -> GPU 0 (UMA read)
    CUDA_CHECK(cudaSetDevice(0));
    test_nocache_bandwidth(
        "Test 2: CPU Memory -> GPU 0 (UMA read)",
        cpu_buf_dev0, gpu0_dst,
        min_chunk_size, max_chunk_size, total_buffer_size,
        iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Test 3: GPU 0 -> CPU Memory (UMA write)
    CUDA_CHECK(cudaSetDevice(0));
    test_nocache_bandwidth(
        "Test 3: GPU 0 -> CPU Memory (UMA write)",
        gpu0_src, cpu_buf_dev0,
        min_chunk_size, max_chunk_size, total_buffer_size,
        iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Multi-GPU tests
    if (deviceCount >= 2 && canAccessPeer) {
        // Test 4: GPU 0 -> GPU 1 (P2P)
        CUDA_CHECK(cudaSetDevice(0));
        test_nocache_bandwidth(
            "Test 4: GPU 0 -> GPU 1 (GPU Direct P2P)",
            gpu0_src, gpu1_dst,
            min_chunk_size, max_chunk_size, total_buffer_size,
            iterations, skip,
            clock_rate_ghz_0, 0);
        
        // Test 5: CPU Memory -> GPU 1 (UMA read)
        CUDA_CHECK(cudaSetDevice(1));
        test_nocache_bandwidth(
            "Test 5: CPU Memory -> GPU 1 (UMA read)",
            cpu_buf_dev1, gpu1_dst,
            min_chunk_size, max_chunk_size, total_buffer_size,
            iterations, skip,
            clock_rate_ghz_1, 1);
    } else if (deviceCount < 2) {
        cout << "\n[Skipping multi-GPU tests: Only 1 GPU available]" << endl;
    } else if (!canAccessPeer) {
        cout << "\n[Skipping multi-GPU tests: P2P not supported]" << endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(gpu0_src));
    CUDA_CHECK(cudaFree(gpu0_dst));
    
    if (deviceCount >= 2 && gpu1_dst != nullptr) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaFree(gpu1_dst));
    }
    
    CUDA_CHECK(cudaFreeHost(cpu_buf));
    
    cout << "\nAll tests completed!" << endl;
    cout << "\nNote: Cache bypass reveals true memory access latency." << endl;
    cout << "Each iteration accesses different chunks from a " << total_buffer_size / 1024 / 1024 << " MB buffer." << endl;
    
    return 0;
}

