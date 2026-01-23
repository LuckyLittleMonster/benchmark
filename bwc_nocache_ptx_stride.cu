/*
 * GPU Bandwidth Test with Hardware-Level Cache Bypass (PTX) - Strided Version
 * 
 * Uses PTX inline assembly with cache modifiers:
 *   - .cv (cache volatile) for loads: bypass cache on read
 *   - .wb (write-back) for stores: bypass cache on write
 * 
 * Tests PUT and GET operations for P2P with strided access:
 *   - Stride: 4B, 8B, 16B, ..., 512B
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
// Supports strided access
__global__ void memcpy_kernel_cv(float* __restrict__ dst, 
                                  const float* __restrict__ src, 
                                  size_t num_floats,
                                  unsigned long long* cycles_out,
                                  int skip_iterations,
                                  int timed_iterations,
                                  int stride_elements) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Warm-up
    for (int iter = 0; iter < skip_iterations; iter++) {
        for (size_t i = tid; i < num_floats; i += block_size) {
            size_t offset = i * stride_elements;
            float val = load_cv(&src[offset]);
            store_wb(&dst[offset], val);
        }
    }
    
    __syncthreads();
    unsigned long long start_cycle = clock64();
    
    // Timed iterations
    for (int iter = 0; iter < timed_iterations; iter++) {
        for (size_t i = tid; i < num_floats; i += block_size) {
            size_t offset = i * stride_elements;
            float val = load_cv(&src[offset]);
            store_wb(&dst[offset], val);
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
                        int device_id,
                        int stride_bytes) {
    
    cout << "\n=== " << test_name << " [Stride: " << stride_bytes << "B] ===" << endl;
    cout << ljust("size(B)", 15) << "\t" 
         << ljust("latency(us)", 15) << "\t" 
         << ljust("bw(MB/s)", 15) << endl;
    
    unsigned long long* d_cycles;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    
    int timed_iterations = iterations - skip;
    int stride_elements = stride_bytes / sizeof(float);
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        size_t num_floats = size / sizeof(float);
        
        int block_size = (num_floats + 31) / 32 * 32; // round up to the nearest multiple of 32
        if (block_size > 1024) block_size = 1024;
        
        // Test .cv+.wb (no cache, true memory latency)
        memcpy_kernel_cv<<<1, block_size>>>(dst, src, num_floats, d_cycles, skip, timed_iterations, stride_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        unsigned long long h_cycles;
        CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        // Calculate metrics
        double total_time_ns = h_cycles / clock_rate_ghz;
        double avg_latency_ns = total_time_ns / timed_iterations;
        double total_time_s = total_time_ns / 1e9;
        
        // Bandwidth calculation:
        // We transferred 'size' bytes of useful data.
        // The actual memory traffic might be higher due to stride, but we usually report effective bandwidth.
        double bandwidth = (size * timed_iterations / (1024.0 * 1024.0)) / total_time_s;
        
        cout << ljust(std::to_string(size), 15) << "\t"
             << ljust(std::to_string(avg_latency_ns / 1000.0), 15) << "\t"
             << ljust(std::to_string(bandwidth), 15) << endl;
    }
    
    CUDA_CHECK(cudaFree(d_cycles));
}

int main() {
    size_t min_size = 4;
    size_t max_size = 128 * 1024;
    int iterations = 32;
    int skip = 3;
    
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
    
    // Calculate total size needed for the largest stride
    // total_size = num_elements * max_stride_bytes
    size_t max_stride = 512;
    size_t total_size = (max_size / sizeof(float)) * max_stride; 
    
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
    
    // Loop through strides
    for (int stride_bytes = 4; stride_bytes <= 512; stride_bytes *= 2) {
        cout << "\n--------------------------------------------------" << endl;
        cout << "Testing Stride: " << stride_bytes << " Bytes" << endl;
        cout << "--------------------------------------------------" << endl;

        // Test 1: GPU 0 -> GPU 0 (Baseline)
        CUDA_CHECK(cudaSetDevice(0));
        test_bandwidth_ptx("Test 1: GPU 0 -> GPU 0 (D2D internal, baseline)",
                           gpu0_src, gpu0_dst,
                           min_size, max_size, iterations, skip,
                           clock_rate_ghz_0, 0, stride_bytes);
        
        // Test 2: CPU -> GPU 0 (UMA read on GPU 0)
        CUDA_CHECK(cudaSetDevice(0));
        test_bandwidth_ptx("Test 2: CPU Memory -> GPU 0 (UMA read, run on GPU 0)",
                           cpu_buf_dev0, gpu0_dst,
                           min_size, max_size, iterations, skip,
                           clock_rate_ghz_0, 0, stride_bytes);
        
        // Test 3: GPU 0 -> CPU (UMA write on GPU 0)
        CUDA_CHECK(cudaSetDevice(0));
        test_bandwidth_ptx("Test 3: GPU 0 -> CPU Memory (UMA write, run on GPU 0)",
                           gpu0_src, cpu_buf_dev0,
                           min_size, max_size, iterations, skip,
                           clock_rate_ghz_0, 0, stride_bytes);
        
        // Multi-GPU tests: PUT and GET
        if (deviceCount >= 2 && canAccessPeer) {
            // Test 4: PUT - GPU 0 writes to GPU 1 (local read, remote write)
            CUDA_CHECK(cudaSetDevice(0));
            test_bandwidth_ptx("Test 4: PUT - GPU 0 -> GPU 1 (run on GPU 0, local read + remote write)",
                               gpu0_src, gpu1_dst,
                               min_size, max_size, iterations, skip,
                               clock_rate_ghz_0, 0, stride_bytes);
            
            // Test 5: GET - GPU 0 reads from GPU 1 (remote read, local write)
            CUDA_CHECK(cudaSetDevice(0));
            test_bandwidth_ptx("Test 5: GET - GPU 1 -> GPU 0 (run on GPU 0, remote read + local write)",
                               gpu1_dst, gpu0_dst,
                               min_size, max_size, iterations, skip,
                               clock_rate_ghz_0, 0, stride_bytes);
        }
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
    
    return 0;
}
