/*
 * Small Message GPU Bandwidth Test with Shared Memory Target
 * 
 * This program tests bandwidth for small messages (4B - 64KB) stored in shared memory.
 * Uses clock64() for high-precision timing inside kernels.
 * 
 * Key difference from bwc.cu: Data is read from various sources but written to 
 * GPU 0's shared memory instead of global memory. This tests the read path performance
 * from different memory locations.
 * 
 * Comparison scenarios:
 * 1. GPU 0 Global -> GPU 0 Shared (Device read to shared, baseline) [Always runs]
 * 2. CPU Memory -> GPU 0 Shared (UMA read to shared) [Always runs]
 * 3. GPU 1 Global -> GPU 0 Shared (P2P read to shared) [Requires 2+ GPUs with P2P support]
 * 
 * Shared memory size limits (A100): ~164KB per SM
 * This test focuses on small messages that fit entirely in shared memory.
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

// Maximum shared memory per block (conservative estimate)
#define MAX_SHARED_SIZE (48 * 1024)  // 48KB, safe for most GPUs

// Kernel that reads from global/CPU/P2P memory and writes to shared memory
__global__ void read_to_shared_kernel(const float* __restrict__ src, 
                                       size_t num_floats,
                                       unsigned long long* cycles_out,
                                       int skip_iterations,
                                       int timed_iterations) {
    // Declare shared memory (dynamically sized)
    extern __shared__ float shared_buf[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Calculate work per thread
    size_t floats_per_thread = (num_floats + block_size - 1) / block_size;
    size_t start_idx = tid * floats_per_thread;
    size_t end_idx = min(start_idx + floats_per_thread, num_floats);
    
    // Warm-up iterations (not timed)
    for (int iter = 0; iter < skip_iterations; iter++) {
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf[i] = src[i];  // Read from src (any memory) -> write to shared
        }
        __syncthreads();
    }
    
    // Synchronize before timing
    __syncthreads();
    
    // Start timing
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf[i] = src[i];  // Read from src -> write to shared
        }
        __syncthreads();
    }
    
    // End timing
    unsigned long long end_cycle = clock64();
    
    // Store cycles (only from thread 0)
    if (tid == 0) {
        cycles_out[0] = end_cycle - start_cycle;
    }
}

// Vectorized version using float4 for better performance
__global__ void read_to_shared_kernel_float4(const float* __restrict__ src, 
                                              size_t num_floats,
                                              unsigned long long* cycles_out,
                                              int skip_iterations,
                                              int timed_iterations) {
    // Declare shared memory (dynamically sized)
    extern __shared__ float shared_buf[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    size_t num_float4 = num_floats / 4;
    float4* shared_buf4 = reinterpret_cast<float4*>(shared_buf);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    
    // Calculate work per thread
    size_t float4_per_thread = (num_float4 + block_size - 1) / block_size;
    size_t start_idx = tid * float4_per_thread;
    size_t end_idx = min(start_idx + float4_per_thread, num_float4);
    
    // Warm-up iterations (not timed)
    for (int iter = 0; iter < skip_iterations; iter++) {
        // Copy float4 chunks
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf4[i] = src4[i];
        }
        
        // Handle remaining floats (only one thread)
        if (tid == 0) {
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < num_floats; i++) {
                shared_buf[i] = src[i];
            }
        }
        __syncthreads();
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        // Copy float4 chunks
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf4[i] = src4[i];
        }
        
        // Handle remaining floats (only one thread)
        if (tid == 0) {
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < num_floats; i++) {
                shared_buf[i] = src[i];
            }
        }
        __syncthreads();
    }
    
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_out[0] = end_cycle - start_cycle;
    }
}

void test_read_to_shared_bandwidth(const string& test_name,
                                    float* src,
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
        // Check if size fits in shared memory
        if (size > MAX_SHARED_SIZE) {
            cout << ljust(std::to_string(size), 15) << "\t"
                 << "[Skipped: exceeds shared memory limit]" << endl;
            continue;
        }
        
        size_t num_floats = size / sizeof(float);
        
        // Use single block with appropriate number of threads
        int block_size = 256;
        if (num_floats < 256) {
            block_size = 32; // Minimum warp size
        }
        
        // Single kernel call with both skip and timed iterations
        // Pass shared memory size as third argument
        if (size >= 16) {
            read_to_shared_kernel_float4<<<1, block_size, size>>>(
                src, num_floats, d_cycles, skip, timed_iterations);
        } else {
            read_to_shared_kernel<<<1, block_size, size>>>(
                src, num_floats, d_cycles, skip, timed_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Read back cycle count
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), 
                              cudaMemcpyDeviceToHost));
        
        // Calculate metrics
        double total_time_ns = h_cycles / clock_rate_ghz;
        double total_time_us = total_time_ns / 1000.0;
        double avg_latency_ns = total_time_ns / timed_iterations;
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
    // Small message sizes: 4B to 48KB (limited by shared memory)
    size_t min_size = 4;
    size_t max_size = 48 * 1024;  // Conservative limit for shared memory
    int iterations = 1000;
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
    
    double clock_rate_ghz_0 = prop0.clockRate / 1e6;
    
    cout << "\nGPU 0: " << prop0.name << endl;
    cout << "  Clock rate: " << clock_rate_ghz_0 << " GHz" << endl;
    cout << "  Shared memory per block: " << prop0.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "  Unified Addressing: " << (prop0.unifiedAddressing ? "Yes" : "No") << endl;
    
    // Update max_size based on actual shared memory
    size_t actual_max_shared = prop0.sharedMemPerBlock;
    if (max_size > actual_max_shared) {
        max_size = actual_max_shared;
        cout << "  Adjusted max test size to: " << max_size / 1024 << " KB" << endl;
    }
    
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
        cout << "\nOnly 1 GPU found. Will skip P2P tests." << endl;
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
    
    // Test 1: GPU 0 Global -> GPU 0 Shared (baseline)
    // Kernel runs on GPU 0, reads from GPU 0 global, writes to GPU 0 shared
    CUDA_CHECK(cudaSetDevice(0));
    test_read_to_shared_bandwidth(
        "Test 1: GPU 0 Global -> GPU 0 Shared (baseline)",
        gpu0_buf,
        min_size, max_size, iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Test 2: CPU Memory -> GPU 0 Shared (UMA read to shared)
    // Kernel runs on GPU 0, reads from CPU memory (UMA), writes to GPU 0 shared
    CUDA_CHECK(cudaSetDevice(0));
    test_read_to_shared_bandwidth(
        "Test 2: CPU Memory -> GPU 0 Shared (UMA read)",
        cpu_buf_dev0,
        min_size, max_size, iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Multi-GPU tests (only if 2+ GPUs available)
    if (deviceCount >= 2 && canAccessPeer) {
        // Test 3: GPU 1 Global -> GPU 0 Shared (P2P read to shared)
        // Kernel runs on GPU 0, reads from GPU 1 global (P2P), writes to GPU 0 shared
        CUDA_CHECK(cudaSetDevice(0));
        test_read_to_shared_bandwidth(
            "Test 3: GPU 1 Global -> GPU 0 Shared (P2P read)",
            gpu1_buf,
            min_size, max_size, iterations, skip,
            clock_rate_ghz_0, 0);
    } else if (deviceCount < 2) {
        cout << "\n[Skipping P2P test: Only 1 GPU available]" << endl;
    } else if (!canAccessPeer) {
        cout << "\n[Skipping P2P test: P2P not supported]" << endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(gpu0_buf));
    
    if (deviceCount >= 2 && gpu1_buf != nullptr) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaFree(gpu1_buf));
    }
    
    CUDA_CHECK(cudaFreeHost(cpu_buf));
    
    cout << "\nAll tests completed!" << endl;
    cout << "\nNote: All data is written to shared memory instead of global memory." << endl;
    cout << "This tests the read performance from different memory sources." << endl;
    
    return 0;
}

