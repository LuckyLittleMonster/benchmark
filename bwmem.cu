/*
 * Comprehensive GPU Bandwidth Test using cudaMemcpy and cudaMemcpyAsync
 * 
 * This program tests 5 different data transfer scenarios:
 * 1. GPU 0 -> GPU 0 (Device-to-Device on same GPU)
 * 2. GPU 0 -> CPU (Device-to-Host)
 * 3. GPU 0 -> GPU 1 with GPU Direct (Direct P2P transfer)
 * 4. GPU 0 -> CPU -> GPU 1 (Staged through CPU, synchronous, no GPU Direct)
 * 5. GPU 0 -> CPU -> GPU 1 (Staged, asynchronous with pipelining)
 *
 * Test 5 uses cudaMemcpyAsync with multiple streams and data chunking to
 * achieve pipelined transfers: while chunk N is being copied D2H, chunk N-1
 * can be copied H2D simultaneously, improving overall throughput.
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

void test_bandwidth(const string& test_name, 
                    void* src, void* dst, void* host_buf,
                    size_t min_size, size_t max_size,
                    int iterations, int skip,
                    cudaMemcpyKind copy_type,
                    bool use_staging = false) {
    
    cout << "\n=== " << test_name << " ===" << endl;
    cout << ljust("size", 30) << "\t" 
         << ljust("time(us)", 30) << "\t" 
         << ljust("bw(MB/s)", 30) << endl;
    
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        float total_time = 0.0;
        
        // Warm-up iterations
        for (int i = 0; i < skip; i++) {
            if (use_staging) {
                // GPU 0 -> Host -> GPU 1
                CUDA_CHECK(cudaMemcpy(host_buf, src, size, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(dst, host_buf, size, cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpy(dst, src, size, copy_type));
            }
        }
        
        // Timed iterations
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        for (int i = 0; i < (iterations - skip); i++) {
            if (use_staging) {
                // GPU 0 -> Host -> GPU 1
                CUDA_CHECK(cudaMemcpy(host_buf, src, size, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(dst, host_buf, size, cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpy(dst, src, size, copy_type));
            }
        }
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start_event, stop_event));
        
        double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0)) / (total_time / 1000.0);
        cout << ljust(std::to_string(size), 30) 
             << ljust(std::to_string(total_time * 1000.0 / (iterations - skip)), 30) 
             << ljust(std::to_string(bandwidth), 30) << endl;
    }
    
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
}

// Async version with chunking for pipelined transfers
void test_bandwidth_async(const string& test_name, 
                          void* src, void* dst, void** host_bufs, int num_chunks,
                          size_t min_size, size_t max_size,
                          int iterations, int skip,
                          cudaMemcpyKind copy_type,
                          bool use_staging = false) {
    
    cout << "\n=== " << test_name << " (Async, " << num_chunks << " chunks) ===" << endl;
    cout << ljust("size", 30) << "\t" 
         << ljust("time(us)", 30) << "\t" 
         << ljust("bw(MB/s)", 30) << endl;
    
    // Create CUDA streams for pipelining
    cudaStream_t* streams = new cudaStream_t[num_chunks];
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        // Skip very small sizes for chunked transfers
        if (use_staging && size < num_chunks * 1024) {
            continue;
        }
        
        size_t chunk_size = size / num_chunks;
        float total_time = 0.0;
        
        // Warm-up iterations
        for (int iter = 0; iter < skip; iter++) {
            if (use_staging) {
                // Pipelined GPU 0 -> Host -> GPU 1
                for (int c = 0; c < num_chunks; c++) {
                    size_t offset = c * chunk_size;
                    CUDA_CHECK(cudaMemcpyAsync((char*)host_bufs[c], (char*)src + offset, 
                                               chunk_size, cudaMemcpyDeviceToHost, streams[c]));
                    CUDA_CHECK(cudaMemcpyAsync((char*)dst + offset, (char*)host_bufs[c], 
                                               chunk_size, cudaMemcpyHostToDevice, streams[c]));
                }
                // Wait for all streams to complete
                for (int c = 0; c < num_chunks; c++) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[c]));
                }
            } else {
                // Chunked async copy
                for (int c = 0; c < num_chunks; c++) {
                    size_t offset = c * chunk_size;
                    CUDA_CHECK(cudaMemcpyAsync((char*)dst + offset, (char*)src + offset, 
                                               chunk_size, copy_type, streams[c]));
                }
                for (int c = 0; c < num_chunks; c++) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[c]));
                }
            }
        }
        
        // Timed iterations
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        for (int iter = 0; iter < (iterations - skip); iter++) {
            if (use_staging) {
                // Pipelined GPU 0 -> Host -> GPU 1
                for (int c = 0; c < num_chunks; c++) {
                    size_t offset = c * chunk_size;
                    CUDA_CHECK(cudaMemcpyAsync((char*)host_bufs[c], (char*)src + offset, 
                                               chunk_size, cudaMemcpyDeviceToHost, streams[c]));
                    CUDA_CHECK(cudaMemcpyAsync((char*)dst + offset, (char*)host_bufs[c], 
                                               chunk_size, cudaMemcpyHostToDevice, streams[c]));
                }
                // Wait for all streams to complete
                for (int c = 0; c < num_chunks; c++) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[c]));
                }
            } else {
                // Chunked async copy
                for (int c = 0; c < num_chunks; c++) {
                    size_t offset = c * chunk_size;
                    CUDA_CHECK(cudaMemcpyAsync((char*)dst + offset, (char*)src + offset, 
                                               chunk_size, copy_type, streams[c]));
                }
                for (int c = 0; c < num_chunks; c++) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[c]));
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start_event, stop_event));
        
        double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0)) / (total_time / 1000.0);
        cout << ljust(std::to_string(size), 30) 
             << ljust(std::to_string(total_time * 1000.0 / (iterations - skip)), 30) 
             << ljust(std::to_string(bandwidth), 30) << endl;
    }
    
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
}

int main(int argc, char *argv[]) {
    size_t min_size = 1;
    size_t max_size = 4ul * 1024 * 1024 * 1024;
    int iterations = 32;
    int skip = 3;
    
    // Check GPU count
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    cout << "Found " << deviceCount << " GPU(s)" << endl;
    
    if (deviceCount < 2) {
        cout << "Warning: Only " << deviceCount << " GPU found. Some P2P tests will be skipped." << endl;
    }
    
    // Get GPU 0 properties
    cudaDeviceProp prop0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
    cout << "GPU 0: " << prop0.name << endl;
    
    if (deviceCount >= 2) {
        cudaDeviceProp prop1;
        CUDA_CHECK(cudaGetDeviceProperties(&prop1, 1));
        cout << "GPU 1: " << prop1.name << endl;
    }
    
    // Allocate memory on GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    void *gpu0_src, *gpu0_dst;
    CUDA_CHECK(cudaMalloc(&gpu0_src, max_size));
    CUDA_CHECK(cudaMalloc(&gpu0_dst, max_size));
    CUDA_CHECK(cudaMemset(gpu0_src, 0, max_size));
    
    // Allocate pinned host memory
    void *host_buf;
    CUDA_CHECK(cudaMallocHost(&host_buf, max_size));
    
    // Allocate multiple host buffers for async chunked transfers
    const int num_chunks = 4;  // Can adjust: 2, 4, 8, etc.
    void** host_bufs = new void*[num_chunks];
    size_t chunk_size = max_size / num_chunks;
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaMallocHost(&host_bufs[i], chunk_size));
    }
    
    // Test 1: GPU 0 -> GPU 0 (Device to Device on same GPU)
    test_bandwidth("Test 1: GPU 0 -> GPU 0 (D2D)", 
                   gpu0_src, gpu0_dst, host_buf,
                   min_size, max_size, iterations, skip,
                   cudaMemcpyDeviceToDevice, false);
    
    // Test 2: GPU 0 -> CPU (Device to Host)
    test_bandwidth("Test 2: GPU 0 -> CPU (D2H)", 
                   gpu0_src, host_buf, nullptr,
                   min_size, max_size, iterations, skip,
                   cudaMemcpyDeviceToHost, false);
    
    // Tests 3 & 4: Only if we have 2+ GPUs
    if (deviceCount >= 2) {
        // Check if GPUs support peer access
        int canAccessPeer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
        
        if (canAccessPeer) {
            cout << "\nGPU 0 can access GPU 1 peer memory" << endl;
            
            // Allocate memory on GPU 1
            CUDA_CHECK(cudaSetDevice(1));
            void *gpu1_dst;
            CUDA_CHECK(cudaMalloc(&gpu1_dst, max_size));
            CUDA_CHECK(cudaMemset(gpu1_dst, 0, max_size));
            
            // Enable peer access for Test 3
            CUDA_CHECK(cudaSetDevice(0));
            cudaError_t peer_err = cudaDeviceEnablePeerAccess(1, 0);
            if (peer_err == cudaSuccess) {
                cout << "GPU Direct enabled for Test 3" << endl;
                
                // Test 3: GPU 0 -> GPU 1 with GPU Direct
                test_bandwidth("Test 3: GPU 0 -> GPU 1 with GPU Direct (P2P)", 
                               gpu0_src, gpu1_dst, host_buf,
                               min_size, max_size, iterations, skip,
                               cudaMemcpyDeviceToDevice, false);
                
                // Disable peer access for Test 4
                CUDA_CHECK(cudaDeviceDisablePeerAccess(1));
                cout << "\nGPU Direct disabled for Test 4" << endl;
            } else if (peer_err == cudaErrorPeerAccessAlreadyEnabled) {
                cout << "GPU Direct already enabled" << endl;
                cudaGetLastError(); // Clear error
                
                // Test 3
                test_bandwidth("Test 3: GPU 0 -> GPU 1 with GPU Direct (P2P)", 
                               gpu0_src, gpu1_dst, host_buf,
                               min_size, max_size, iterations, skip,
                               cudaMemcpyDeviceToDevice, false);
                
                // Try to disable for Test 4
                cudaDeviceDisablePeerAccess(1);
            }
            
            // Test 4: GPU 0 -> CPU -> GPU 1 (no GPU Direct, synchronous)
            test_bandwidth("Test 4: GPU 0 -> CPU -> GPU 1 (Staged, no GPU Direct, Sync)", 
                           gpu0_src, gpu1_dst, host_buf,
                           min_size, max_size, iterations, skip,
                           cudaMemcpyDefault, true);
            
            // Test 5: GPU 0 -> CPU -> GPU 1 (async with chunking for pipelining)
            test_bandwidth_async("Test 5: GPU 0 -> CPU -> GPU 1 (Staged, Async with Pipelining)", 
                                 gpu0_src, gpu1_dst, host_bufs, num_chunks,
                                 min_size, max_size, iterations, skip,
                                 cudaMemcpyDefault, true);
            
            // Cleanup GPU 1
            CUDA_CHECK(cudaSetDevice(1));
            CUDA_CHECK(cudaFree(gpu1_dst));
        } else {
            cout << "\nWarning: GPU 0 cannot access GPU 1 peer memory. Skipping P2P tests." << endl;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(gpu0_src));
    CUDA_CHECK(cudaFree(gpu0_dst));
    CUDA_CHECK(cudaFreeHost(host_buf));
    
    // Free chunked host buffers
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaFreeHost(host_bufs[i]));
    }
    delete[] host_bufs;
    
    cout << "\nAll tests completed!" << endl;
    
    return 0;
}

