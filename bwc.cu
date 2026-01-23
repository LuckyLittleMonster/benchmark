/*
 * Small Message GPU Bandwidth Test with High-Precision Cycle Timer
 * 
 * This program tests bandwidth for small messages (4B - 64KB) that can fit in a single block.
 * Uses clock64() for high-precision timing inside kernels.
 * 
 * Comparison scenarios:
 * 1. GPU 0 -> GPU 0 (Device-to-Device internal, baseline) [Always runs]
 * 2. CPU Memory -> GPU 0 (GPU accessing CPU via UMA read) [Always runs]
 * 3. GPU 0 -> CPU Memory (GPU accessing CPU via UMA write) [Always runs]
 * 4. GPU 0 -> GPU 1 (GPU Direct P2P) [Requires 2+ GPUs with P2P support]
 * 5. CPU Memory -> GPU 1 (GPU accessing CPU via UMA read) [Requires 2+ GPUs with P2P support]
 * 6. GPU 0 -> CPU -> GPU 1 (two-hop through CPU memory staging) [Requires 2+ GPUs with P2P support]
 * 
 * If only 1 GPU is available, tests 4-6 will be skipped.
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

// Kernel with internal cycle-based timing for small messages
// Each thread copies a portion of data and measures cycles
// Performs skip iterations first (warm-up), then timed iterations
__global__ void memcpy_kernel_with_timing(float* __restrict__ dst, 
                                           const float* __restrict__ src, 
                                           size_t num_floats,
                                           unsigned long long* cycles_per_thread,
                                           int skip_iterations,
                                           int timed_iterations) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Calculate work per thread
    size_t floats_per_thread = (num_floats + block_size - 1) / block_size;
    size_t start_idx = tid * floats_per_thread;
    size_t end_idx = min(start_idx + floats_per_thread, num_floats);
    
    // Warm-up iterations (not timed)
    for (int iter = 0; iter < skip_iterations; iter++) {
        for (size_t i = start_idx; i < end_idx; i++) {
            dst[i] = src[i];
        }
    }
    
    // Synchronize before timing
    __syncthreads();
    
    // Start timing
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        for (size_t i = start_idx; i < end_idx; i++) {
            dst[i] = src[i];
        }
    }
    
    // End timing
    __syncthreads();
    unsigned long long end_cycle = clock64();
    
    // Store cycles (only from thread 0 to avoid variance)
    if (tid == 0) {
        cycles_per_thread[0] = end_cycle - start_cycle;
    }
}

// Vectorized version using float4 for better performance
__global__ void memcpy_kernel_float4_with_timing(float* __restrict__ dst, 
                                                   const float* __restrict__ src, 
                                                   size_t num_floats,
                                                   unsigned long long* cycles_per_thread,
                                                   int skip_iterations,
                                                   int timed_iterations) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    size_t num_float4 = num_floats / 4;
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    
    // Calculate work per thread
    size_t float4_per_thread = (num_float4 + block_size - 1) / block_size;
    size_t start_idx = tid * float4_per_thread;
    size_t end_idx = min(start_idx + float4_per_thread, num_float4);
    
    // Warm-up iterations (not timed)
    for (int iter = 0; iter < skip_iterations; iter++) {
        // Copy float4 chunks
        for (size_t i = start_idx; i < end_idx; i++) {
            dst4[i] = src4[i];
        }
        
        // Handle remaining floats (only one thread)
        if (tid == 0) {
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < num_floats; i++) {
                dst[i] = src[i];
            }
        }
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        // Copy float4 chunks
        for (size_t i = start_idx; i < end_idx; i++) {
            dst4[i] = src4[i];
        }
        
        // Handle remaining floats (only one thread)
        if (tid == 0) {
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < num_floats; i++) {
                dst[i] = src[i];
            }
        }
    }
    
    __syncthreads();
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_per_thread[0] = end_cycle - start_cycle;
    }
}

void test_small_message_bandwidth(const string& test_name,
                                   float* src, float* dst,
                                   size_t min_size, size_t max_size,
                                   int iterations, int skip,
                                   double clock_rate_ghz,
                                   int device_id) {
    
    cout << "\n=== " << test_name << " ===" << endl;
    cout << ljust("size(B)", 15) << "\t" 
         << ljust("cycles", 18) << "\t"
         << ljust("avg_lat(ns)", 15) << "\t" 
         << ljust("total_time(us)", 18) << "\t"
         << ljust("bw(GB/s)", 15) << endl;
    
    // Allocate cycle counter on GPU
    unsigned long long* d_cycles;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    
    unsigned long long h_cycles;
    int timed_iterations = iterations - skip;
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        size_t num_floats = size / sizeof(float);
        
        // Use single block with appropriate number of threads
        int block_size = 256;
        if (num_floats < 256) {
            block_size = 32; // Minimum warp size
        }
        
        // Single kernel call with both skip and timed iterations
        if (size >= 16) {
            memcpy_kernel_float4_with_timing<<<1, block_size>>>(
                dst, src, num_floats, d_cycles, skip, timed_iterations);
        } else {
            memcpy_kernel_with_timing<<<1, block_size>>>(
                dst, src, num_floats, d_cycles, skip, timed_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Read back cycle count
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), 
                              cudaMemcpyDeviceToHost));
        
        // Calculate metrics
        // Total time for all timed iterations
        double total_time_ns = h_cycles / clock_rate_ghz; // Convert cycles to nanoseconds
        double total_time_us = total_time_ns / 1000.0; // Convert to microseconds
        
        // Average latency per single iteration
        double avg_latency_ns = total_time_ns / timed_iterations;
        
        // Bandwidth (total data transferred / total time)
        double total_time_s = total_time_ns / 1e9;
        double bandwidth = (size * timed_iterations / (1024.0 * 1024.0 * 1024.0)) / total_time_s;
        
        cout << ljust(std::to_string(size), 15) << "\t"
             << ljust(std::to_string(h_cycles), 18) << "\t"
             << ljust(std::to_string(avg_latency_ns), 15) << "\t"
             << ljust(std::to_string(total_time_us), 18) << "\t"
             << ljust(std::to_string(bandwidth), 15) << endl;
    }
    
    CUDA_CHECK(cudaFree(d_cycles));
}

int main(int argc, char *argv[]) {
    // Small message sizes: 4B to 64KB
    size_t min_size = 4;
    size_t max_size = 64 * 1024;
    int iterations = 1000; // More iterations for small messages
    int skip = 10;
    
    // Check GPU count
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    cout << "Found " << deviceCount << " GPU(s)" << endl;
    
    if (deviceCount < 1) {
        cout << "Error: Need at least 1 GPU for this test" << endl;
        return 1;
    }
    
    // Get GPU properties and clock rates
    cudaDeviceProp prop0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
    
    double clock_rate_ghz_0 = prop0.clockRate / 1e6; // Convert kHz to GHz
    
    cout << "\nGPU 0: " << prop0.name << endl;
    cout << "  Clock rate: " << clock_rate_ghz_0 << " GHz" << endl;
    cout << "  Unified Addressing: " << (prop0.unifiedAddressing ? "Yes" : "No") << endl;
    
    double clock_rate_ghz_1 = 0.0;
    bool canAccessPeer = false;
    
    if (deviceCount >= 2) {
        cudaDeviceProp prop1;
        CUDA_CHECK(cudaGetDeviceProperties(&prop1, 1));
        clock_rate_ghz_1 = prop1.clockRate / 1e6;
        
        cout << "\nGPU 1: " << prop1.name << endl;
        cout << "  Clock rate: " << clock_rate_ghz_1 << " GHz" << endl;
        cout << "  Unified Addressing: " << (prop1.unifiedAddressing ? "Yes" : "No") << endl;
        
        // Check P2P capability
        int canAccessPeerInt;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeerInt, 0, 1));
        canAccessPeer = (canAccessPeerInt != 0);
        cout << "\nP2P Access (GPU 0 -> GPU 1): " << (canAccessPeer ? "Supported" : "Not supported") << endl;
    } else {
        cout << "\nOnly 1 GPU found. Will skip multi-GPU tests." << endl;
    }
    
    // Allocate memory on GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    float *gpu0_buf;
    CUDA_CHECK(cudaMalloc(&gpu0_buf, max_size));
    CUDA_CHECK(cudaMemset(gpu0_buf, 0, max_size));
    
    // Allocate memory on GPU 1 (only if available)
    float *gpu1_buf = nullptr;
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMalloc(&gpu1_buf, max_size));
        CUDA_CHECK(cudaMemset(gpu1_buf, 0, max_size));
    }
    
    // Allocate CPU pinned memory (for UMA access from GPU)
    float *cpu_buf;
    CUDA_CHECK(cudaHostAlloc(&cpu_buf, max_size, cudaHostAllocMapped));
    memset(cpu_buf, 0, max_size);
    
    // Get device pointers for the host memory (for UMA access)
    float *cpu_buf_dev0, *cpu_buf_dev1 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaHostGetDevicePointer(&cpu_buf_dev0, cpu_buf, 0));
    
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaHostGetDevicePointer(&cpu_buf_dev1, cpu_buf, 0));
    }
    
    // Enable P2P access from GPU 0 to GPU 1 (only if available and supported)
    if (deviceCount >= 2 && canAccessPeer) {
        CUDA_CHECK(cudaSetDevice(0));
        cudaError_t peer_err = cudaDeviceEnablePeerAccess(1, 0);
        if (peer_err == cudaSuccess) {
            cout << "\nP2P enabled: GPU 0 -> GPU 1\n" << endl;
        } else if (peer_err == cudaErrorPeerAccessAlreadyEnabled) {
            cout << "\nP2P already enabled: GPU 0 -> GPU 1\n" << endl;
            cudaGetLastError(); // Clear error
        } else {
            CUDA_CHECK(peer_err);
        }
    }
    
    // Allocate second buffer on GPU 0 for internal test
    CUDA_CHECK(cudaSetDevice(0));
    float *gpu0_buf2;
    CUDA_CHECK(cudaMalloc(&gpu0_buf2, max_size));
    CUDA_CHECK(cudaMemset(gpu0_buf2, 0, max_size));
    
    // Test 1: GPU 0 -> GPU 0 (internal, baseline)
    CUDA_CHECK(cudaSetDevice(0));
    test_small_message_bandwidth(
        "Test 1: GPU 0 -> GPU 0 (D2D internal, baseline)",
        gpu0_buf, gpu0_buf2,
        min_size, max_size, iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Test 2: CPU Memory -> GPU 0 (GPU accessing CPU via UMA)
    // Kernel runs on GPU 0, reads from CPU memory (UMA), writes to GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    test_small_message_bandwidth(
        "Test 2: CPU Memory -> GPU 0 (UMA read)",
        cpu_buf_dev0, gpu0_buf,
        min_size, max_size, iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Test 3: GPU 0 -> CPU Memory (GPU accessing CPU via UMA)
    // Kernel runs on GPU 0, reads from GPU 0, writes to CPU memory (UMA)
    CUDA_CHECK(cudaSetDevice(0));
    test_small_message_bandwidth(
        "Test 3: GPU 0 -> CPU Memory (UMA write)",
        gpu0_buf, cpu_buf_dev0,
        min_size, max_size, iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Multi-GPU tests (only if 2+ GPUs available)
    if (deviceCount >= 2 && canAccessPeer) {
        // Test 4: GPU 0 -> GPU 1 (GPU Direct P2P)
        // Kernel runs on GPU 0, reads from GPU 0, writes to GPU 1
        CUDA_CHECK(cudaSetDevice(0));
        test_small_message_bandwidth(
            "Test 4: GPU 0 -> GPU 1 (GPU Direct P2P)",
            gpu0_buf, gpu1_buf,
            min_size, max_size, iterations, skip,
            clock_rate_ghz_0, 0);
        
        // Test 5: CPU Memory -> GPU 1 (GPU accessing CPU via UMA)
        // Kernel runs on GPU 1, reads from CPU memory, writes to GPU 1
        CUDA_CHECK(cudaSetDevice(1));
        test_small_message_bandwidth(
            "Test 5: CPU Memory -> GPU 1 (UMA read)",
            cpu_buf_dev1, gpu1_buf,
            min_size, max_size, iterations, skip,
            clock_rate_ghz_1, 1);
        
        // Test 6: GPU 0 -> CPU Memory -> GPU 1 (two-hop)
        // Note: This shows the two separate hops for reference
        // First copy: GPU 0 -> CPU (kernel on GPU 0)
        cout << "\n=== Test 6a: GPU 0 -> CPU (part of two-hop transfer) ===" << endl;
        CUDA_CHECK(cudaSetDevice(0));
        test_small_message_bandwidth(
            "Test 6a: GPU 0 -> CPU Memory",
            gpu0_buf, cpu_buf_dev0,
            min_size, max_size, iterations, skip,
            clock_rate_ghz_0, 0);
        
        // Second copy: CPU -> GPU 1 (kernel on GPU 1)
        cout << "\n=== Test 6b: CPU -> GPU 1 (part of two-hop transfer) ===" << endl;
        CUDA_CHECK(cudaSetDevice(1));
        test_small_message_bandwidth(
            "Test 6b: CPU Memory -> GPU 1",
            cpu_buf_dev1, gpu1_buf,
            min_size, max_size, iterations, skip,
            clock_rate_ghz_1, 1);
    } else if (deviceCount < 2) {
        cout << "\n[Skipping multi-GPU tests: Only 1 GPU available]" << endl;
    } else if (!canAccessPeer) {
        cout << "\n[Skipping multi-GPU tests: P2P not supported]" << endl;
    }
    
    // Cleanup GPU 0 second buffer
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(gpu0_buf2));
    
    // Cleanup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(gpu0_buf));
    
    if (deviceCount >= 2 && gpu1_buf != nullptr) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaFree(gpu1_buf));
    }
    
    CUDA_CHECK(cudaFreeHost(cpu_buf));
    
    cout << "\nAll tests completed!" << endl;
    
    return 0;
}

