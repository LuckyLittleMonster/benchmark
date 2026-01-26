#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <string>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <semaphore.h>
#include <cstring>
using namespace std;

string ljust(string s, int width) {
    return s + string(width - s.size(), ' ');
}

#define CUDA_CHECK(cmd) do {                          \
    cudaError_t e = cmd;                               \
    if( e != cudaSuccess ) {                            \
        printf("Failed: Cuda error %s:%d '%s'\n",        \
                __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                \
    }                                                       \
} while(0)

#define NCCL_CHECK(cmd) do {                          \
    ncclResult_t r = cmd;                              \
    if( r != ncclSuccess ) {                            \
        printf("Failed: NCCL error %s:%d '%s'\n",        \
                __FILE__,__LINE__,ncclGetErrorString(r)); \
        exit(EXIT_FAILURE);                                \
    }                                                       \
} while(0)

// Shared memory structure for IPC communication
struct SharedData {
    ncclUniqueId nccl_id;
    sem_t sem_sync;  // Synchronization barrier
    sem_t sem_parent_done;
    sem_t sem_cleanup_sync;  // Synchronization for cleanup
    int parent_done;
};

// CUDA kernel for one-sided data movement using device-side NCCL LSA
__global__ void memcpy_kernel_float4(ncclDevComm devComm, ncclWindow_t win, size_t num_floats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t num_float4 = num_floats / 4;
    
    // Get pointers to local and remote memory
    float* local_ptr = (float*)ncclGetLocalPointer(win, 0);
    float* peer_ptr = (float*)ncclGetLsaPointer(win, 0, 1); // peer rank 1
    
    float4* dst4 = reinterpret_cast<float4*>(peer_ptr);
    const float4* src4 = reinterpret_cast<const float4*>(local_ptr);
    
    // Vectorized copy using float4
    for (size_t i = idx; i < num_float4; i += stride) {
        dst4[i] = src4[i];
    }
    
    // Handle remaining elements
    size_t remaining_start = num_float4 * 4;
    for (size_t i = remaining_start + idx; i < num_floats; i += stride) {
        peer_ptr[i] = local_ptr[i];
    }
}

// Simple kernel for small sizes
__global__ void memcpy_kernel_simple(ncclDevComm devComm, ncclWindow_t win, size_t num_floats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float* local_ptr = (float*)ncclGetLocalPointer(win, 0);
    float* remote_ptr = (float*)ncclGetLsaPointer(win, 0, 1);
    
    for (size_t i = idx; i < num_floats; i += stride) {
        remote_ptr[i] = local_ptr[i];
    }
}

void child_process(SharedData* shared) {
    int device = 1;
    CUDA_CHECK(cudaSetDevice(device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cout << "Child process using GPU " << device << ": " << prop.name << endl;
    
    size_t max_size = 4ul * 1024 * 1024 * 1024;
    
    // Signal that child is ready
    sem_post(&shared->sem_sync);
    
    // Initialize NCCL communicator for GPU 1 (collective)
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, 2, shared->nccl_id, 1));
    
    // Allocate symmetric memory using NCCL
    void *gpu_ptr;
    NCCL_CHECK(ncclMemAlloc(&gpu_ptr, max_size));
    CUDA_CHECK(cudaMemset(gpu_ptr, 0, max_size));
    
    // Register window for symmetric access (collective)
    ncclWindow_t window;
    NCCL_CHECK(ncclCommWindowRegister(comm, gpu_ptr, max_size, &window, NCCL_WIN_COLL_SYMMETRIC));
    
    // Create device communicator (COLLECTIVE - must be called by both processes)
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    reqs.lsaBarrierCount = 1;
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    
    cout << "Child: NCCL comm initialized, waiting for parent to finish..." << endl;
    
    // Wait for parent to finish testing
    sem_wait(&shared->sem_parent_done);
    
    cout << "Child: Parent finished, starting cleanup..." << endl;
    
    // Signal ready for cleanup
    sem_post(&shared->sem_cleanup_sync);
    
    // Cleanup (all collective operations)
    NCCL_CHECK(ncclDevCommDestroy(comm, &devComm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    NCCL_CHECK(ncclMemFree(gpu_ptr));
    NCCL_CHECK(ncclCommDestroy(comm));
    
    cout << "Child: Cleanup complete" << endl;
}

void parent_process(SharedData* shared) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cout << "Parent process using GPU " << device << ": " << prop.name << endl;
    
    size_t min_size = 1;
    size_t max_size = 4ul * 1024 * 1024 * 1024;
    int iterations = 32;
    int skip = 3;
    
    // Wait for child to be ready
    sem_wait(&shared->sem_sync);
    
    // Initialize NCCL communicator for GPU 0 (collective)
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, 2, shared->nccl_id, 0));
    
    // Allocate symmetric memory using NCCL
    void *local_ptr;
    NCCL_CHECK(ncclMemAlloc(&local_ptr, max_size));
    CUDA_CHECK(cudaMemset(local_ptr, 0, max_size));
    
    // Register window for symmetric access
    ncclWindow_t window;
    NCCL_CHECK(ncclCommWindowRegister(comm, local_ptr, max_size, &window, NCCL_WIN_COLL_SYMMETRIC));
    
    // Create device communicator
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    reqs.lsaBarrierCount = 1;
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    
    // Check if GPUs support peer access
    int canAccessPeer;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (!canAccessPeer) {
        cout << "Error: GPU 0 cannot access GPU 1" << endl;
        shared->parent_done = 1;
        sem_post(&shared->sem_parent_done);
        exit(1);
    }
    
    // Enable peer access
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    
    cout << "\n=== P2P Bandwidth Test (GPU 0 -> GPU 1 via NCCL) ===" << endl;
    cout << ljust("size", 30) << "\t" 
         << ljust("time(ms)", 30) << "\t" 
         << ljust("bw(GB/s)", 30) << endl;
    
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    for (size_t size = min_size; size <= max_size; size *= 2) {
        float total_time = 0.0;
        size_t num_floats = size / sizeof(float);
        
        // Calculate grid and block dimensions
        int blockSize = 256;
        int numBlocks = (num_floats + blockSize - 1) / blockSize;
        numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
        
        // Warm-up iterations
        for (int i = 0; i < skip; i++) {
            if (size >= 16) {
                memcpy_kernel_float4<<<numBlocks, blockSize>>>(devComm, window, num_floats);
            } else {
                memcpy_kernel_simple<<<1, 32>>>(devComm, window, num_floats);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Record start time
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        for (int i = 0; i < (iterations - skip); i++) {
            if (size >= 16) {
                memcpy_kernel_float4<<<numBlocks, blockSize>>>(devComm, window, num_floats);
            } else {
                memcpy_kernel_simple<<<1, 32>>>(devComm, window, num_floats);
            }
        }
        // Record stop time
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start_event, stop_event));
        
        double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
        cout << ljust(to_string(size), 30) << ljust(to_string(total_time), 30) << ljust(to_string(bandwidth), 30) << endl;
    }
    
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    
    cout << "\nParent: Testing complete, starting cleanup..." << endl;
    
    // Signal child that we're done testing
    shared->parent_done = 1;
    sem_post(&shared->sem_parent_done);
    
    // Wait for child to be ready for cleanup
    sem_wait(&shared->sem_cleanup_sync);
    
    // Cleanup (all collective operations)
    NCCL_CHECK(ncclDevCommDestroy(comm, &devComm));
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    NCCL_CHECK(ncclMemFree(local_ptr));
    NCCL_CHECK(ncclCommDestroy(comm));
    
    cout << "Parent: Cleanup complete" << endl;
}

int main(int argc, char *argv[]) {
    // Create shared memory for NCCL unique ID
    const char* shm_name = "/nccl_ipc_shm";
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return 1;
    }
    
    // Set size of shared memory
    if (ftruncate(shm_fd, sizeof(SharedData)) == -1) {
        perror("ftruncate");
        return 1;
    }
    
    // Map shared memory
    SharedData* shared = (SharedData*)mmap(NULL, sizeof(SharedData), 
                                           PROT_READ | PROT_WRITE, 
                                           MAP_SHARED, shm_fd, 0);
    if (shared == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    
    // Initialize semaphores
    sem_init(&shared->sem_sync, 1, 0);
    sem_init(&shared->sem_parent_done, 1, 0);
    sem_init(&shared->sem_cleanup_sync, 1, 0);
    shared->parent_done = 0;
    
    // Generate NCCL unique ID in main process before fork
    NCCL_CHECK(ncclGetUniqueId(&shared->nccl_id));
    
    // Fork process
    pid_t pid = fork();
    
    if (pid < 0) {
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // Child process
        child_process(shared);
    } else {
        // Parent process
        parent_process(shared);
        
        // Wait for child to finish
        wait(NULL);
        
        cout << "\nBoth processes completed successfully!" << endl;
    }
    
    // Cleanup shared memory
    sem_destroy(&shared->sem_sync);
    sem_destroy(&shared->sem_parent_done);
    sem_destroy(&shared->sem_cleanup_sync);
    munmap(shared, sizeof(SharedData));
    
    if (pid != 0) {
        // Only parent unlinks
        shm_unlink(shm_name);
    }
    
    return 0;
}