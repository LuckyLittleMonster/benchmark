#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
using namespace std;

/*
 * Device-Side Intra-Node NCCL GPU Direct RDMA Bandwidth & Latency Test
 * Format matches bwl.c and nccl_bwl.c for easy comparison
 */

#define MAX_MSG_SIZE (10UL * 1024 * 1024)  // 10 MB
#define WARMUP_ITERS 3
#define BANDWIDTH_ITERS 29
#define LATENCY_ITERS 29

static int rank;

// Error Checking Macros
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
    } \
} while(0)

#define CHECK_NCCL(call) do { \
    ncclResult_t res = (call); \
    if (res != ncclSuccess) { \
        cerr << "NCCL Error: " << ncclGetErrorString(res) << endl; \
    } \
} while(0)

struct GPUContext {
    int device_id;
    ncclComm_t comm;
    ncclDevComm devComm;
    cudaStream_t stream;
    void* buffer;
    ncclWindow_t window;
    cudaEvent_t start, stop;
};

void setup_gpu(GPUContext& ctx, int nGPUs, const ncclUniqueId& id) {
    CHECK_CUDA(cudaSetDevice(ctx.device_id));
    
    // Initialize NCCL communicator
    CHECK_NCCL(ncclCommInitRank(&ctx.comm, nGPUs, id, ctx.device_id));
    
    // Allocate symmetric memory using NCCL
    CHECK_NCCL(ncclMemAlloc(&ctx.buffer, MAX_MSG_SIZE));
    CHECK_CUDA(cudaMemset(ctx.buffer, ctx.device_id, MAX_MSG_SIZE));
    
    // Register window for symmetric access
    CHECK_NCCL(ncclCommWindowRegister(ctx.comm, ctx.buffer, MAX_MSG_SIZE, 
                                       &ctx.window, NCCL_WIN_COLL_SYMMETRIC));
    
    // Create device communicator
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    reqs.lsaBarrierCount = 1;  // 1 CTA per GPU
    CHECK_NCCL(ncclDevCommCreate(ctx.comm, &reqs, &ctx.devComm));
    
    // Create stream and events
    CHECK_CUDA(cudaStreamCreate(&ctx.stream));
    CHECK_CUDA(cudaEventCreate(&ctx.start));
    CHECK_CUDA(cudaEventCreate(&ctx.stop));
}

void cleanup_gpu(GPUContext& ctx) {
    CHECK_CUDA(cudaSetDevice(ctx.device_id));
    CHECK_CUDA(cudaEventDestroy(ctx.start));
    CHECK_CUDA(cudaEventDestroy(ctx.stop));
    CHECK_CUDA(cudaStreamDestroy(ctx.stream));
    CHECK_NCCL(ncclDevCommDestroy(ctx.comm, &ctx.devComm));
    CHECK_NCCL(ncclCommWindowDeregister(ctx.comm, ctx.window));
    CHECK_NCCL(ncclMemFree(ctx.buffer));
    CHECK_NCCL(ncclCommDestroy(ctx.comm));
}

// Device-side kernel using LSA (Load/Store Accessible) for direct memory access
template<typename T>
__global__ void unidir_lsa_kernel(ncclDevComm devComm, 
                                   ncclWindow_t win,
                                   size_t count,
                                   int iters) {
    // Create barrier for synchronization across GPUs
    ncclLsaBarrierSession<ncclCoopCta> bar {
        ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x
    };
    
    const int my_rank = devComm.lsaRank;
    const int peer = (my_rank == 0) ? 1 : 0;
    
    // Get pointers to local and peer memory
    T* local_ptr = (T*)ncclGetLocalPointer(win, 0);
    T* peer_ptr = (T*)ncclGetLsaPointer(win, 0, peer);
    
    for (int iter = 0; iter < iters; iter++) {
        // Pre-iteration barrier
        bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
        
        // Only rank 0 writes to rank 1
        if (my_rank == 0) {
            for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
                peer_ptr[i] = local_ptr[i];
            }
        }
        
        // Post-iteration barrier ensures write completion
        bar.sync(ncclCoopCta(), cuda::memory_order_release);
    }
}

/*
 * Test 1: Unidirectional bandwidth (GPU0 -> GPU1) using device-side NCCL
 * Uses CUDA events for timing
 */
void test_unidirectional(vector<GPUContext>& contexts, size_t msg_size) {
    size_t count = msg_size / sizeof(float);
    int nCTAs = 1;
    int nThreads = 256;
    
    // Launch warmup on both GPUs
    for (auto& ctx : contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        unidir_lsa_kernel<float><<<nCTAs, nThreads, 0, ctx.stream>>>(
            ctx.devComm, ctx.window, count, WARMUP_ITERS);
    }
    
    // Synchronize both GPUs
    for (auto& ctx : contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
    }
    
    // Start timing on both GPUs
    for (auto& ctx : contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        CHECK_CUDA(cudaEventRecord(ctx.start, ctx.stream));
    }
    
    // Launch timed kernels
    for (auto& ctx : contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        unidir_lsa_kernel<float><<<nCTAs, nThreads, 0, ctx.stream>>>(
            ctx.devComm, ctx.window, count, BANDWIDTH_ITERS);
    }
    
    // Record end events
    for (auto& ctx : contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        CHECK_CUDA(cudaEventRecord(ctx.stop, ctx.stream));
    }
    
    // Wait for completion
    float max_elapsed_ms = 0.0f;
    for (auto& ctx : contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        CHECK_CUDA(cudaEventSynchronize(ctx.stop));
        
        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ctx.start, ctx.stop));
        max_elapsed_ms = max(max_elapsed_ms, elapsed_ms);
    }
    
    // Calculate and output results
    double elapsed_s = max_elapsed_ms / 1000.0;
    double latency_us = (elapsed_s / BANDWIDTH_ITERS) * 1e6;
    double bw_mbps = (static_cast<double>(msg_size) * BANDWIDTH_ITERS) /
                     (elapsed_s * 1024.0 * 1024.0);
    
    cout << left
         << setw(30) << msg_size
         << setw(30) << latency_us
         << setw(30) << bw_mbps
         << endl;
}

int main(int argc, char **argv) {
    int nGPUs;
    CHECK_CUDA(cudaGetDeviceCount(&nGPUs));
    
    if (nGPUs < 2) {
        cerr << "This test requires at least 2 GPUs" << endl;
        return -1;
    }
    
    // Use only 2 GPUs
    nGPUs = 2;
    
    cout << "=== Device-Side NCCL GPU Direct Performance Tests ===" << endl;
    cout << "Using LSA (Load/Store Accessible) communication" << endl;
    cout << "Max message size: " << (MAX_MSG_SIZE / (1024 * 1024)) << " MB" << endl;
    cout << endl;
    
    // Get unique ID for NCCL
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    
    // Setup GPU contexts
    vector<GPUContext> contexts(nGPUs);
    vector<thread> threads;
    
    for (int i = 0; i < nGPUs; i++) {
        contexts[i].device_id = i;
        threads.emplace_back([&, i]() {
            setup_gpu(contexts[i], nGPUs, id);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
    
    // Test 1: Unidirectional
    cout << "=== Test 1: Unidirectional (GPU 0 -> GPU 1) ===" << endl;
    cout << left
         << setw(30) << "size"
         << setw(30) << "latency(us)"
         << setw(30) << "BW(MB/s)" << endl;
    
    for (size_t msg_size = sizeof(float); msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test_unidirectional(contexts, msg_size);
    }
    
    cout << endl << "=== All Tests Complete ===" << endl;
    
    // Cleanup
    for (int i = 0; i < nGPUs; i++) {
        cleanup_gpu(contexts[i]);
    }
    
    return 0;
}