/*
 * MPI GPU Direct Bandwidth & Latency Test
 * Format matches bwl.c for easy comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define MAX_MSG_SIZE (10UL * 1024 * 1024)  // 10 MB
#define WARMUP_ITERS 3
#define BANDWIDTH_ITERS 29
#define LATENCY_ITERS 29

static int rank, size;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA Error: %s\n", rank, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

/*
 * Test 1: Unidirectional bandwidth (Rank 0 -> Rank 1)
 */
static void test1_unidirectional(void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test
    for (int i = 0; i < BANDWIDTH_ITERS; i++) {
        if (rank == 0) {
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    if (rank == 0) {
        elapsed = end - start;
        double latency_us = (elapsed / BANDWIDTH_ITERS) * 1000000.0;
        double bw_mbps = msg_size / elapsed / BANDWIDTH_ITERS / (1024.0 * 1024.0);
        printf("%-30zu%-30.6f%-30.6f\n", msg_size, latency_us, bw_mbps);
        fflush(stdout);
    }
}

/*
 * Test 2: Ping-pong with 1-byte ack
 */
static void test2_pingpong_ack(void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(gpu_buf, 1, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(gpu_buf, 1, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test
    for (int i = 0; i < LATENCY_ITERS; i++) {
        if (rank == 0) {
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(gpu_buf, 1, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(gpu_buf, 1, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    if (rank == 0) {
        elapsed = end - start;
        double latency_us = (elapsed / LATENCY_ITERS) * 1000000.0;
        double total_data = (double)msg_size + 1.0;
        double bw_mbps = total_data / elapsed / LATENCY_ITERS / (1024.0 * 1024.0);
        printf("%-30zu%-30.6f%-30.6f\n", msg_size, latency_us, bw_mbps);
        fflush(stdout);
    }
}

/*
 * Test 3: Ping-pong with same-size reply
 */
static void test3_pingpong_same(void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test
    for (int i = 0; i < LATENCY_ITERS; i++) {
        if (rank == 0) {
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(gpu_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(gpu_buf, msg_size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    if (rank == 0) {
        elapsed = end - start;
        double latency_us = (elapsed / LATENCY_ITERS / 2.0) * 1000000.0;
        double total_data = (double)msg_size * 2.0;
        double bw_mbps = total_data / elapsed / LATENCY_ITERS / (1024.0 * 1024.0);
        printf("%-30zu%-30.6f%-30.6f\n", msg_size, latency_us, bw_mbps);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly 2 processes\n");
        }
        MPI_Finalize();
        return -1;
    }
    
    // Set GPU device
    int gpu_id = 0;
    CHECK_CUDA(cudaSetDevice(gpu_id));
    
    if (rank == 0) {
        printf("=== MPI GPU Direct RDMA Performance Tests ===\n");
        printf("Using GPU-aware MPI (Cray MPICH)\n");
        printf("Max message size: %lu MB\n\n", MAX_MSG_SIZE / (1024 * 1024));
    }
    
    // Allocate GPU memory
    void *gpu_buf;
    CHECK_CUDA(cudaMalloc(&gpu_buf, MAX_MSG_SIZE));
    CHECK_CUDA(cudaMemset(gpu_buf, 0, MAX_MSG_SIZE));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 1: Unidirectional
    if (rank == 0) {
        printf("=== Test 1: Unidirectional (Rank 0 -> Rank 1) ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test1_unidirectional(gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 2: Ping-pong with 1-byte ack
    if (rank == 0) {
        printf("=== Test 2: Ping-Pong with 1-byte Ack ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test2_pingpong_ack(gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 3: Ping-pong with same-size reply
    if (rank == 0) {
        printf("=== Test 3: Ping-Pong with Same-size Reply ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test3_pingpong_same(gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n=== All Tests Complete ===\n");
    }
    
    // Cleanup
    cudaFree(gpu_buf);
    MPI_Finalize();
    
    return 0;
}

