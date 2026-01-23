/*
 * GPU Bandwidth Test with Hardware-Level Cache Bypass (PTX)
 * 
 * Uses PTX inline assembly with cache modifiers:
 *   - .cv (cache volatile) for loads: bypass cache on read
 *   - .wb (write-back) for stores: bypass cache on write
 * 
 * Tests PUT and GET operations for P2P:
 *   - PUT: Run on GPU 0, local read + remote write to GPU 1
 *   - GET: Run on GPU 0, remote read from GPU 1 + local write
 * 
 * All multi-GPU tests run on GPU 0 to measure from single GPU perspective.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <string>

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

// Device function to load with .cv (cache volatile) - bypasses cache
__device__ __forceinline__ float load_cv(const float* addr) {
    float value;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(value) : "l"(addr));
    return value;
}

// Device function to store with .wb (write-back) - bypasses cache for write
__device__ __forceinline__ void store_wb(float* addr, float value) {
    asm volatile("st.global.wb.f32 [%0], %1;" :: "l"(addr), "f"(value));
}


// Kernel using .cv for load (bypass cache) and .wb for store
__global__ void memcpy_kernel_cv(float* __restrict__ dst, 
                                  const float* __restrict__ src, 
                                  size_t num_floats,
                                  unsigned long long* cycles_out,
                                  int skip_iterations,
                                  int timed_iterations) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Warm-up
    for (int iter = 0; iter < skip_iterations; iter++) {
        for (size_t i = tid; i < num_floats; i += block_size) {
            float val = load_cv(&src[i]);
            store_wb(&dst[i], val);
        }
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        for (size_t i = tid; i < num_floats; i += block_size) {
            float val = load_cv(&src[i]);
            store_wb(&dst[i], val);
        }
    }
    
    __syncthreads();
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_out[0] = end_cycle - start_cycle;
    }
}


void test_bandwidth_ptx(const string& test_name,
                        float* src, float* dst,
                        size_t min_size, size_t max_size,
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
    
    int timed_iterations = iterations - skip;
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        size_t num_floats = size / sizeof(float);
        
        // int block_size = 256;
        // if (num_floats < 256) block_size = 32;
        int block_size = (num_floats + 31) / 32 * 32; // round up to the nearest multiple of 32
        if (block_size > 1024) block_size = 1024;
        
        // Test .cv+.wb (no cache, true memory latency)
        memcpy_kernel_cv<<<1, block_size>>>(dst, src, num_floats, d_cycles, skip, timed_iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long h_cycles;
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        // Calculate metrics (same format as bwc.cu)
        double total_time_ns = h_cycles / clock_rate_ghz;
        double total_time_us = total_time_ns / 1000.0;
        double avg_latency_ns = total_time_ns / timed_iterations;
        double total_time_s = total_time_ns / 1e9;
        double bandwidth = (size * timed_iterations / (1024.0 * 1024.0)) / total_time_s;
        
        cout << ljust(std::to_string(size), 15) << "\t"
            //  << ljust(std::to_string(h_cycles), 18) << "\t"
             << ljust(std::to_string(avg_latency_ns / 1000.0), 15) << "\t"
            //  << ljust(std::to_string(total_time_us), 18) << "\t"
             << ljust(std::to_string(bandwidth), 15) << endl;
    }
    
    CUDA_CHECK(cudaFree(d_cycles));
}

int main() {
    size_t min_size = 4;
    size_t max_size = 64 * 1024;
    int iterations = 1000;
    int skip = 10;
    
    // Check GPU count
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    cout << "Found " << deviceCount << " GPU(s)" << endl;
    
    cudaDeviceProp prop0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
    double clock_rate_ghz_0 = prop0.clockRate / 1e6;
    
    cout << "\nGPU 0: " << prop0.name << endl;
    cout << "  Clock rate: " << clock_rate_ghz_0 << " GHz" << endl;
    cout << "  L2 Cache: " << prop0.l2CacheSize / 1024 / 1024 << " MB" << endl;
    
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
    
    cout << "\nUsing PTX cache bypass instructions:" << endl;
    cout << "  ld.global.cv.f32 - load with cache volatile (bypasses cache on read)" << endl;
    cout << "  st.global.wb.f32 - store with write-back (bypasses cache on write)" << endl;
    cout << "\nP2P Test Types:" << endl;
    cout << "  PUT: GPU 0 kernel reads local memory, writes to GPU 1 (local read + remote write)" << endl;
    cout << "  GET: GPU 0 kernel reads GPU 1 memory, writes local (remote read + local write)" << endl;
    
    size_t total_size = 4 * 1024 * 1024;  // 4MB
    
    // GPU 0 buffers
    CUDA_CHECK(cudaSetDevice(0));
    float *gpu0_src, *gpu0_dst;
    CUDA_CHECK(cudaMalloc(&gpu0_src, total_size));
    CUDA_CHECK(cudaMalloc(&gpu0_dst, total_size));
    CUDA_CHECK(cudaMemset(gpu0_src, 0, total_size));
    CUDA_CHECK(cudaMemset(gpu0_dst, 0, total_size));
    
    // GPU 1 buffer (if available)
    float *gpu1_dst = nullptr;
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMalloc(&gpu1_dst, total_size));
        CUDA_CHECK(cudaMemset(gpu1_dst, 0, total_size));
    }
    
    // CPU buffer
    float *cpu_buf, *cpu_buf_dev0, *cpu_buf_dev1 = nullptr;
    CUDA_CHECK(cudaHostAlloc(&cpu_buf, total_size, cudaHostAllocMapped));
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
    
    // Test 1: GPU 0 -> GPU 0 (Baseline)
    CUDA_CHECK(cudaSetDevice(0));
    test_bandwidth_ptx("Test 1: GPU 0 -> GPU 0 (D2D internal, baseline)",
                       gpu0_src, gpu0_dst,
                       min_size, max_size, iterations, skip,
                       clock_rate_ghz_0, 0);
    
    // Test 2: CPU -> GPU 0 (UMA read on GPU 0)
    CUDA_CHECK(cudaSetDevice(0));
    test_bandwidth_ptx("Test 2: CPU Memory -> GPU 0 (UMA read, run on GPU 0)",
                       cpu_buf_dev0, gpu0_dst,
                       min_size, max_size, iterations, skip,
                       clock_rate_ghz_0, 0);
    
    // Test 3: GPU 0 -> CPU (UMA write on GPU 0)
    CUDA_CHECK(cudaSetDevice(0));
    test_bandwidth_ptx("Test 3: GPU 0 -> CPU Memory (UMA write, run on GPU 0)",
                       gpu0_src, cpu_buf_dev0,
                       min_size, max_size, iterations, skip,
                       clock_rate_ghz_0, 0);
    
    // Multi-GPU tests: PUT and GET
    if (deviceCount >= 2 && canAccessPeer) {
        // Test 4: PUT - GPU 0 writes to GPU 1 (local read, remote write)
        CUDA_CHECK(cudaSetDevice(0));
        test_bandwidth_ptx("Test 4: PUT - GPU 0 -> GPU 1 (run on GPU 0, local read + remote write)",
                           gpu0_src, gpu1_dst,
                           min_size, max_size, iterations, skip,
                           clock_rate_ghz_0, 0);
        
        // Test 5: GET - GPU 0 reads from GPU 1 (remote read, local write)
        CUDA_CHECK(cudaSetDevice(0));
        test_bandwidth_ptx("Test 5: GET - GPU 1 -> GPU 0 (run on GPU 0, remote read + local write)",
                           gpu1_dst, gpu0_dst,
                           min_size, max_size, iterations, skip,
                           clock_rate_ghz_0, 0);
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
    cout << "\nNote: Using PTX .cv (load) + .wb (store) for complete cache bypass." << endl;
    cout << "This measures true memory access latency without cache." << endl;
    
    return 0;
}

