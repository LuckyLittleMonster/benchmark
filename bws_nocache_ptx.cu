/*
 * Shared Memory Bandwidth Test with Hardware-Level Cache Bypass (PTX)
 * 
 * Uses PTX inline assembly with cache modifiers for reads:
 *   - .cv (cache volatile) for loads: bypass cache on read
 * 
 * Data is read from various sources and written to shared memory,
 * allowing us to measure pure read performance.
 * 
 * For P2P, only GET is tested (shared memory cannot do PUT):
 *   - GET: Run on GPU 0, remote read from GPU 1 to local shared memory
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

#define MAX_SHARED_SIZE (48 * 1024)

// Device function to load with .cv (cache volatile) - bypasses cache
__device__ __forceinline__ float load_cv(const float* addr) {
    float value;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(value) : "l"(addr));
    return value;
}


// Kernel using .cv (no cache) for reading to shared memory
__global__ void read_to_shared_cv(const float* __restrict__ src, 
                                   size_t num_floats,
                                   unsigned long long* cycles_out,
                                   int skip_iterations,
                                   int timed_iterations) {
    extern __shared__ float shared_buf[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Warm-up
    for (int iter = 0; iter < skip_iterations; iter++) {
        for (size_t i = tid; i < num_floats; i += block_size) {
            shared_buf[i] = load_cv(&src[i]);
        }
        __syncthreads();
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        for (size_t i = tid; i < num_floats; i += block_size) {
            shared_buf[i] = load_cv(&src[i]);
        }
        __syncthreads();
    }
    // __syncthreads();
    unsigned long long end_cycle = clock64();
    
    if (tid == 0) {
        cycles_out[0] = end_cycle - start_cycle;
    }
}


void test_shared_bandwidth_ptx(const string& test_name,
                                float* src,
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
        if (size > MAX_SHARED_SIZE) {
            cout << ljust(std::to_string(size), 15) << "\t"
                 << "[Skipped: exceeds shared memory limit]" << endl;
            continue;
        }
        
        size_t num_floats = size / sizeof(float);
        
        // int block_size = 256;
        // if (num_floats < 256) block_size = 32;
        int block_size = (num_floats + 31) / 32 * 32; // round up to the nearest multiple of 32
        if (block_size > 1024) block_size = 1024;
        // Test .cv (no cache) read to shared memory
        read_to_shared_cv<<<1, block_size, size>>>(src, num_floats, d_cycles, skip, timed_iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long h_cycles;
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        // Calculate metrics (same format as bws.cu)
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
    size_t max_size = 48 * 1024;  // Limited by shared memory
    int iterations = 1000;
    int skip = 10;
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    cout << "Found " << deviceCount << " GPU(s)" << endl;
    
    cudaDeviceProp prop0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
    double clock_rate_ghz_0 = prop0.clockRate / 1e6;
    
    cout << "\nGPU 0: " << prop0.name << endl;
    cout << "  Clock rate: " << clock_rate_ghz_0 << " GHz" << endl;
    cout << "  L2 Cache: " << prop0.l2CacheSize / 1024 / 1024 << " MB" << endl;
    cout << "  Shared Memory per Block: " << prop0.sharedMemPerBlock / 1024 << " KB" << endl;
    
    // Adjust max_size based on actual shared memory
    size_t actual_max_shared = prop0.sharedMemPerBlock;
    if (max_size > actual_max_shared) {
        max_size = actual_max_shared;
        cout << "  Adjusted max size to: " << max_size / 1024 << " KB" << endl;
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
        cout << "\nOnly 1 GPU found. Will skip multi-GPU tests." << endl;
    }
    
    cout << "\nUsing PTX cache bypass instructions for reads to shared memory:" << endl;
    cout << "  ld.global.cv.f32 - load with cache volatile (bypasses cache on read)" << endl;
    cout << "  Data written to: Shared Memory" << endl;
    cout << "\nP2P Test Type:" << endl;
    cout << "  GET only: GPU 0 kernel reads from GPU 1, writes to local shared memory" << endl;
    cout << "  (PUT is not applicable for shared memory target)" << endl;
    
    size_t total_size = 64 * 1024 * 1024;  // 64MB
    
    // GPU 0 buffer
    float *gpu0_buf;
    CUDA_CHECK(cudaMalloc(&gpu0_buf, total_size));
    CUDA_CHECK(cudaMemset(gpu0_buf, 0, total_size));
    
    // GPU 1 buffer (if available)
    float *gpu1_buf = nullptr;
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMalloc(&gpu1_buf, total_size));
        CUDA_CHECK(cudaMemset(gpu1_buf, 0, total_size));
    }
    
    // CPU buffer (mapped to both GPUs)
    float *cpu_buf, *cpu_buf_dev_0, *cpu_buf_dev_1 = nullptr;
    CUDA_CHECK(cudaHostAlloc(&cpu_buf, total_size, cudaHostAllocMapped));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaHostGetDevicePointer(&cpu_buf_dev_0, cpu_buf, 0));
    
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaHostGetDevicePointer(&cpu_buf_dev_1, cpu_buf, 0));
    }
    
    // Enable P2P if available
    if (deviceCount >= 2 && canAccessPeer) {
        CUDA_CHECK(cudaSetDevice(0));
        cudaError_t peer_err = cudaDeviceEnablePeerAccess(1, 0);
        if (peer_err == cudaSuccess || peer_err == cudaErrorPeerAccessAlreadyEnabled) {
            if (peer_err == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            cout << "P2P enabled: GPU 0 -> GPU 1\n" << endl;
        }
    }
    
    // Test 1: GPU 0 Global -> GPU 0 Shared (Baseline)
    CUDA_CHECK(cudaSetDevice(0));
    test_shared_bandwidth_ptx("Test 1: GPU 0 Global -> GPU 0 Shared (baseline, run on GPU 0)",
                              gpu0_buf,
                              min_size, max_size, iterations, skip,
                              clock_rate_ghz_0, 0);
    
    // Test 2: CPU -> GPU 0 Shared (UMA read on GPU 0)
    CUDA_CHECK(cudaSetDevice(0));
    test_shared_bandwidth_ptx("Test 2: CPU Memory -> GPU 0 Shared (UMA read, run on GPU 0)",
                              cpu_buf_dev_0,
                              min_size, max_size, iterations, skip,
                              clock_rate_ghz_0, 0);
    
    // Multi-GPU tests: GET only (shared memory cannot be used for PUT)
    if (deviceCount >= 2 && canAccessPeer) {
        // Test 3: GET - GPU 0 reads from GPU 1 to shared memory (remote read to local shared mem)
        CUDA_CHECK(cudaSetDevice(0));
        test_shared_bandwidth_ptx("Test 3: GET - GPU 1 -> GPU 0 Shared (run on GPU 0, remote read to local shared)",
                                  gpu1_buf,
                                  min_size, max_size, iterations, skip,
                                  clock_rate_ghz_0, 0);
    } else if (deviceCount < 2) {
        cout << "\n[Skipping multi-GPU tests: Only 1 GPU available]" << endl;
    } else if (!canAccessPeer) {
        cout << "\n[Skipping P2P tests: P2P not supported between GPU 0 and GPU 1]" << endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(gpu0_buf));
    if (gpu1_buf) {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaFree(gpu1_buf));
    }
    CUDA_CHECK(cudaFreeHost(cpu_buf));
    
    cout << "\nAll tests completed!" << endl;
    cout << "\nNote: Using PTX .cv for complete cache bypass on reads to shared memory." << endl;
    cout << "This measures true read latency without cache interference." << endl;
    
    return 0;
}

