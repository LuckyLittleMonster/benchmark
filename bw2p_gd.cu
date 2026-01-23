#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <semaphore.h>
#include <cstring>

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

// Shared memory structure for IPC communication
struct SharedData {
    cudaIpcMemHandle_t ipc_handle;
    sem_t sem_child_ready;  // Child signals when IPC handle is ready
    sem_t sem_parent_done;  // Parent signals when test is done
    int child_ready;
    int parent_done;
};

// CUDA kernel for data movement using vectorized load/store
__global__ void memcpy_kernel_float4(float* __restrict__ dst, const float* __restrict__ src, size_t num_floats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t num_float4 = num_floats / 4;
    
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    
    // Vectorized copy using float4
    for (size_t i = idx; i < num_float4; i += stride) {
        dst4[i] = src4[i];
    }
    
    // Handle remaining elements (if num_floats is not divisible by 4)
    size_t remaining_start = num_float4 * 4;
    for (size_t i = remaining_start + idx; i < num_floats; i += stride) {
        dst[i] = src[i];
    }
}

// Simple kernel for small sizes
__global__ void memcpy_kernel_simple(float* __restrict__ dst, const float* __restrict__ src, size_t num_floats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_floats; i += stride) {
        dst[i] = src[i];
    }
}

void child_process(SharedData* shared) {
    int device = 1;
    CUDA_CHECK(cudaSetDevice(device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cout << "Child process using GPU " << device << ": " << prop.name << endl;
    
    size_t max_size = 4ul * 1024 * 1024 * 1024;
    
    // Allocate memory on GPU 1
    float *gpu_ptr;
    CUDA_CHECK(cudaMalloc(&gpu_ptr, max_size));
    CUDA_CHECK(cudaMemset(gpu_ptr, 0, max_size));
    
    // Create IPC handle
    CUDA_CHECK(cudaIpcGetMemHandle(&shared->ipc_handle, gpu_ptr));
    
    // Signal parent that IPC handle is ready
    shared->child_ready = 1;
    sem_post(&shared->sem_child_ready);
    
    cout << "Child: IPC handle created, waiting for parent to finish..." << endl;
    
    // Wait for parent to finish testing
    sem_wait(&shared->sem_parent_done);
    
    cout << "Child: Parent finished, cleaning up..." << endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(gpu_ptr));
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
    
    // Allocate memory on GPU 0
    float *local_ptr;
    CUDA_CHECK(cudaMalloc(&local_ptr, max_size));
    CUDA_CHECK(cudaMemset(local_ptr, 0, max_size));
    
    // Wait for child to create IPC handle
    cout << "Parent: Waiting for child to create IPC handle..." << endl;
    sem_wait(&shared->sem_child_ready);
    
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
    
    // Open IPC handle to access child's GPU memory
    float *remote_ptr;
    CUDA_CHECK(cudaIpcOpenMemHandle((void**)&remote_ptr, shared->ipc_handle, cudaIpcMemLazyEnablePeerAccess));
    
    cout << "\n=== P2P Bandwidth Test (GPU 0 -> GPU 1 via fork+IPC) ===" << endl;
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
        // Limit number of blocks for very large transfers
        numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
        
        // Warm-up iterations
        for (int i = 0; i < skip; i++) {
            if (size >= 16) {  // Use vectorized kernel for sizes >= 16 bytes
                memcpy_kernel_float4<<<numBlocks, blockSize>>>(remote_ptr, local_ptr, num_floats);
            } else {
                memcpy_kernel_simple<<<1, 32>>>(remote_ptr, local_ptr, num_floats);
            }
        }
        
        // Record start time before the loop
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        for (int i = 0; i < (iterations - skip); i++) {
            if (size >= 16) {  // Use vectorized kernel for sizes >= 16 bytes
                memcpy_kernel_float4<<<numBlocks, blockSize>>>(remote_ptr, local_ptr, num_floats);
            } else {
                memcpy_kernel_simple<<<1, 32>>>(remote_ptr, local_ptr, num_floats);
            }
        }
        // Record stop time after the loop
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start_event, stop_event));
        
        double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
        cout << ljust(std::to_string(size), 30) << ljust(std::to_string(total_time), 30) << ljust(std::to_string(bandwidth), 30) << endl;
    }
    
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    
    // Close IPC handle
    CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
    
    // Cleanup
    CUDA_CHECK(cudaFree(local_ptr));
    
    // Signal child that we're done
    shared->parent_done = 1;
    sem_post(&shared->sem_parent_done);
}

int main(int argc, char *argv[]) {
    // Create shared memory for IPC
    const char* shm_name = "/cuda_ipc_shm";
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
    sem_init(&shared->sem_child_ready, 1, 0);  // pshared=1 for inter-process
    sem_init(&shared->sem_parent_done, 1, 0);
    shared->child_ready = 0;
    shared->parent_done = 0;
    
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
    sem_destroy(&shared->sem_child_ready);
    sem_destroy(&shared->sem_parent_done);
    munmap(shared, sizeof(SharedData));
    
    if (pid != 0) {
        // Only parent unlinks
        shm_unlink(shm_name);
    }
    
    return 0;
}
