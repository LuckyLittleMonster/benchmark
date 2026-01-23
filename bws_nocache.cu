/*
 * Small Message GPU Bandwidth Test with Shared Memory - Cache Bypass Version
 * 
 * This program tests bandwidth for reading to shared memory with L2 cache bypass.
 * Each iteration reads from different source chunks to avoid cache hits.
 * Uses clock64() for high-precision timing inside kernels.
 * 
 * Comparison scenarios:
 * 1. GPU 0 Global -> GPU 0 Shared (Device read to shared, baseline) [Always runs]
 * 2. CPU Memory -> GPU 0 Shared (UMA read to shared) [Always runs]
 * 3. GPU 1 Global -> GPU 0 Shared (P2P read to shared) [Requires 2+ GPUs with P2P support]
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

#define MAX_SHARED_SIZE (48 * 1024)  // 48KB safe limit

// Read from different source chunks to shared memory (cache bypass)
__global__ void read_to_shared_nocache(const float* __restrict__ src, 
                                        size_t chunk_size,
                                        int num_chunks,
                                        unsigned long long* cycles_out,
                                        int skip_iterations,
                                        int timed_iterations) {
    extern __shared__ float shared_buf[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    size_t floats_per_thread = (chunk_size + block_size - 1) / block_size;
    size_t start_idx = tid * floats_per_thread;
    size_t end_idx = min(start_idx + floats_per_thread, chunk_size);
    
    // Warm-up: access different chunks
    for (int iter = 0; iter < skip_iterations; iter++) {
        size_t chunk_offset = (iter % num_chunks) * chunk_size;
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf[i] = src[chunk_offset + i];
        }
        __syncthreads();
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed: access different chunks each iteration
    for (int iter = 0; iter < timed_iterations; iter++) {
        size_t chunk_offset = ((skip_iterations + iter) % num_chunks) * chunk_size;
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf[i] = src[chunk_offset + i];
        }
        __syncthreads();
    }
    
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_out[0] = end_cycle - start_cycle;
    }
}

// Vectorized version using float4
__global__ void read_to_shared_nocache_float4(const float* __restrict__ src, 
                                               size_t chunk_size,
                                               int num_chunks,
                                               unsigned long long* cycles_out,
                                               int skip_iterations,
                                               int timed_iterations) {
    extern __shared__ float shared_buf[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    size_t num_float4 = chunk_size / 4;
    float4* shared_buf4 = reinterpret_cast<float4*>(shared_buf);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    
    size_t float4_per_thread = (num_float4 + block_size - 1) / block_size;
    size_t start_idx = tid * float4_per_thread;
    size_t end_idx = min(start_idx + float4_per_thread, num_float4);
    
    // Warm-up
    for (int iter = 0; iter < skip_iterations; iter++) {
        size_t chunk_offset = (iter % num_chunks) * num_float4;
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf4[i] = src4[chunk_offset + i];
        }
        
        if (tid == 0) {
            size_t base = (iter % num_chunks) * chunk_size;
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < chunk_size; i++) {
                shared_buf[i] = src[base + i];
            }
        }
        __syncthreads();
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed
    for (int iter = 0; iter < timed_iterations; iter++) {
        size_t chunk_offset = ((skip_iterations + iter) % num_chunks) * num_float4;
        for (size_t i = start_idx; i < end_idx; i++) {
            shared_buf4[i] = src4[chunk_offset + i];
        }
        
        if (tid == 0) {
            size_t base = ((skip_iterations + iter) % num_chunks) * chunk_size;
            size_t remaining_start = num_float4 * 4;
            for (size_t i = remaining_start; i < chunk_size; i++) {
                shared_buf[i] = src[base + i];
            }
        }
        __syncthreads();
    }
    
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_out[0] = end_cycle - start_cycle;
    }
}

void test_read_to_shared_nocache(const string& test_name,
                                  float* src,
                                  size_t min_chunk_size, size_t max_chunk_size,
                                  size_t total_buffer_size,
                                  int iterations, int skip,
                                  double clock_rate_ghz,
                                  int device_id) {
    
    cout << "\n=== " << test_name << " ===" << endl;
    cout << ljust("size(B)", 15) << "\t" 
        //  << ljust("cycles", 18) << "\t" 
         << ljust("latency(us)", 15) << "\t" 
        //  << ljust("total_time(us)", 18) << "\t"
         << ljust("bw(MB/s)", 15) << endl;
    
    unsigned long long* d_cycles;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    
    unsigned long long h_cycles;
    int timed_iterations = iterations - skip;
    
    for (size_t chunk_size = min_chunk_size; chunk_size <= max_chunk_size; chunk_size *= 2) {
        if (chunk_size > MAX_SHARED_SIZE) {
            cout << ljust(std::to_string(chunk_size), 15) << "\t"
                 << "[Skipped: exceeds shared memory limit]" << endl;
            continue;
        }
        
        // Calculate number of chunks
        int num_chunks = total_buffer_size / chunk_size;
        if (num_chunks < 10) num_chunks = 10;
        
        size_t num_floats = chunk_size / sizeof(float);
        
        int block_size = 256;
        if (num_floats < 256) {
            block_size = 32;
        }
        
        // Launch with dynamic shared memory
        if (chunk_size >= 16) {
            read_to_shared_nocache_float4<<<1, block_size, chunk_size>>>(
                src, num_floats, num_chunks, d_cycles, skip, timed_iterations);
        } else {
            read_to_shared_nocache<<<1, block_size, chunk_size>>>(
                src, num_floats, num_chunks, d_cycles, skip, timed_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), 
                              cudaMemcpyDeviceToHost));
        
        // Calculate metrics
        double total_time_ns = h_cycles / clock_rate_ghz;
        double total_time_us = total_time_ns / 1000.0;
        double avg_latency_ns = total_time_ns / timed_iterations;
        double total_time_s = total_time_ns / 1e9;
        double bandwidth = (chunk_size * timed_iterations / (1024.0 * 1024.0)) / total_time_s;
        
        cout << ljust(std::to_string(chunk_size), 15) << "\t"
            //  << ljust(std::to_string(h_cycles), 18) << "\t"
             << ljust(std::to_string(avg_latency_ns / 1000.0), 15) << "\t"
            //  << ljust(std::to_string(total_time_us), 18) << "\t"
             << ljust(std::to_string(bandwidth), 15) << endl;
    }
    
    CUDA_CHECK(cudaFree(d_cycles));
}

int main(int argc, char *argv[]) {
    // Test parameters
    size_t min_chunk_size = 4;           // 4B
    size_t max_chunk_size = 48 * 1024;   // 48KB (limited by shared memory)
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
    cout << "  Shared memory per block: " << prop0.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "  Unified Addressing: " << (prop0.unifiedAddressing ? "Yes" : "No") << endl;
    
    // Adjust max_chunk_size based on actual shared memory
    size_t actual_max_shared = prop0.sharedMemPerBlock;
    if (max_chunk_size > actual_max_shared) {
        max_chunk_size = actual_max_shared;
        cout << "  Adjusted max chunk size to: " << max_chunk_size / 1024 << " KB" << endl;
    }
    
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
        cout << "\nOnly 1 GPU found. Will skip P2P tests." << endl;
    }
    
    // Allocate large buffer on GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    float *gpu0_buf;
    CUDA_CHECK(cudaMalloc(&gpu0_buf, total_buffer_size));
    CUDA_CHECK(cudaMemset(gpu0_buf, 0, total_buffer_size));
    
    // Allocate large buffer on GPU 1 (if available)
    float *gpu1_buf = nullptr;
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMalloc(&gpu1_buf, total_buffer_size));
        CUDA_CHECK(cudaMemset(gpu1_buf, 0, total_buffer_size));
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
    
    // Test 1: GPU 0 Global -> GPU 0 Shared (baseline)
    CUDA_CHECK(cudaSetDevice(0));
    test_read_to_shared_nocache(
        "Test 1: GPU 0 Global -> GPU 0 Shared (baseline)",
        gpu0_buf,
        min_chunk_size, max_chunk_size, total_buffer_size,
        iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Test 2: CPU Memory -> GPU 0 Shared (UMA read)
    CUDA_CHECK(cudaSetDevice(0));
    test_read_to_shared_nocache(
        "Test 2: CPU Memory -> GPU 0 Shared (UMA read)",
        cpu_buf_dev0,
        min_chunk_size, max_chunk_size, total_buffer_size,
        iterations, skip,
        clock_rate_ghz_0, 0);
    
    // Multi-GPU tests
    if (deviceCount >= 2 && canAccessPeer) {
        // Test 3: GPU 1 Global -> GPU 0 Shared (P2P read)
        CUDA_CHECK(cudaSetDevice(0));
        test_read_to_shared_nocache(
            "Test 3: GPU 1 Global -> GPU 0 Shared (P2P read)",
            gpu1_buf,
            min_chunk_size, max_chunk_size, total_buffer_size,
            iterations, skip,
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
    cout << "\nNote: Cache bypass reveals true read performance to shared memory." << endl;
    cout << "Each iteration reads from different source chunks in a " << total_buffer_size / 1024 / 1024 << " MB buffer." << endl;
    
    return 0;
}
