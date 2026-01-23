/*
 * NCCL Point-to-Point GPU Direct RDMA Test
 * 
 * 使用NCCL进行GPU Direct RDMA性能测试
 * NCCL已经解决了所有底层libfabric初始化问题
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define WARMUP_ITERS 10
#define TEST_ITERS 100
#define MAX_SIZE (16 * 1024 * 1024)

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed: NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void test_nccl_bandwidth(ncclComm_t comm, int rank, void* sendbuf, void* recvbuf, 
                         size_t size, cudaStream_t stream) {
    double start, end, elapsed;
    int peer = (rank + 1) % 2;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            NCCLCHECK(ncclSend(sendbuf, size, ncclChar, peer, comm, stream));
            NCCLCHECK(ncclRecv(recvbuf, size, ncclChar, peer, comm, stream));
        } else {
            NCCLCHECK(ncclRecv(recvbuf, size, ncclChar, peer, comm, stream));
            NCCLCHECK(ncclSend(sendbuf, size, ncclChar, peer, comm, stream));
        }
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test
    for (int i = 0; i < TEST_ITERS; i++) {
        if (rank == 0) {
            NCCLCHECK(ncclSend(sendbuf, size, ncclChar, peer, comm, stream));
            NCCLCHECK(ncclRecv(recvbuf, size, ncclChar, peer, comm, stream));
        } else {
            NCCLCHECK(ncclRecv(recvbuf, size, ncclChar, peer, comm, stream));
            NCCLCHECK(ncclSend(sendbuf, size, ncclChar, peer, comm, stream));
        }
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    
    end = MPI_Wtime();
    elapsed = end - start;
    
    if (rank == 0) {
        double total_bytes = 2.0 * TEST_ITERS * size;
        double bandwidth = (total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);
        printf("%10zu bytes: %.2f GB/s\n", size, bandwidth);
    }
}

int main(int argc, char* argv[]) {
    int rank, nranks;
    int device_count;
    ncclComm_t comm;
    ncclUniqueId id;
    void *sendbuf, *recvbuf;
    cudaStream_t stream;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Get_processor_name(processor_name, &name_len);
    
    if (nranks != 2) {
        if (rank == 0) {
            printf("This test requires exactly 2 ranks\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Initialize CUDA
    CUDACHECK(cudaGetDeviceCount(&device_count));
    CUDACHECK(cudaSetDevice(rank % device_count));
    
    struct cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, rank % device_count));
    
    printf("Rank %d on %s using GPU: %s\n", rank, processor_name, prop.name);
    
    // Create CUDA stream
    CUDACHECK(cudaStreamCreate(&stream));
    
    // Allocate GPU memory
    CUDACHECK(cudaMalloc(&sendbuf, MAX_SIZE));
    CUDACHECK(cudaMalloc(&recvbuf, MAX_SIZE));
    CUDACHECK(cudaMemset(sendbuf, rank, MAX_SIZE));
    CUDACHECK(cudaMemset(recvbuf, 0, MAX_SIZE));
    
    // Initialize NCCL
    // NCCL会自动处理所有底层libfabric初始化
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    
    if (rank == 0) {
        printf("\n=== NCCL GPU Direct RDMA Bandwidth Test ===\n");
        printf("Using NCCL for GPU-to-GPU communication\n");
        printf("NCCL handles all libfabric/CXI initialization internally\n\n");
        printf("Message Size    Bandwidth\n");
        printf("------------    ---------\n");
    }
    
    // Test different message sizes
    for (size_t size = 1024; size <= MAX_SIZE; size *= 2) {
        test_nccl_bandwidth(comm, rank, sendbuf, recvbuf, size, stream);
    }
    
    // Cleanup
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaFree(sendbuf));
    CUDACHECK(cudaFree(recvbuf));
    CUDACHECK(cudaStreamDestroy(stream));
    
    if (rank == 0) {
        printf("\n=== Test Complete ===\n");
        printf("Note: NCCL automatically uses the optimal transport\n");
        printf("      (likely libfabric CXI provider on Perlmutter)\n");
    }
    
    MPI_Finalize();
    return 0;
}

