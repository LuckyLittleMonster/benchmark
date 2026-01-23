/*
 * GPU Direct RDMA Bandwidth and Latency Test
 * 
 * Strategy: MPI coordination + Independent libfabric instance (NCCL's approach)
 * - Use MPI for process coordination and address exchange
 * - Create independent libfabric instance (not reusing MPI's)
 * - Use libfabric for actual GPU data transfer with GPU Direct RDMA
 * - Measure bandwidth and latency with optimized polling (no sleep)
 * 
 * Output format matches bw.log
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include <cuda_runtime.h>
#include <sys/uio.h>

#define MAX_MSG_SIZE (1UL * 1024 * 1024 * 1024)  // 2 GB max (CXI limit)
#define WARMUP_ITERS 3
#define BANDWIDTH_ITERS 29
#define LATENCY_ITERS 29

// Tight polling timeout (iterations, not time)
#define POLL_TIMEOUT 10000000

#define CHECK_FI(call, msg) do { \
    int ret = (call); \
    if (ret != 0) { \
        fprintf(stderr, "[Rank %d] Error %s: %s (%d)\n", rank, msg, fi_strerror(-ret), ret); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA Error: %s\n", rank, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

typedef struct {
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_ep *ep;
    struct fid_cq *cq;
    struct fid_av *av;
    struct fi_info *info;
    fi_addr_t peer_addr;
} ofi_context_t;

static int rank = 0;

/*
 * Wait for CQ completion with tight polling (no sleep)
 * This is critical for low latency
 */
static int wait_for_completion(struct fid_cq *cq) {
    struct fi_cq_entry entry;
    int ret;
    int poll_count = 0;
    
    while (poll_count++ < POLL_TIMEOUT) {
        ret = fi_cq_read(cq, &entry, 1);
        if (ret > 0) {
            return 0;  // Success
        } else if (ret < 0 && ret != -FI_EAGAIN) {
            struct fi_cq_err_entry err_entry;
            fi_cq_readerr(cq, &err_entry, 0);
            fprintf(stderr, "[Rank %d] CQ error: %s\n", rank, fi_strerror(err_entry.err));
            return -1;
        }
        // No sleep - tight polling for best performance
    }
    
    fprintf(stderr, "[Rank %d] CQ timeout after %d polls\n", rank, POLL_TIMEOUT);
    return -1;
}

/*
 * Initialize libfabric with independent instance (NCCL's approach)
 */
static int init_ofi(ofi_context_t *ctx) {
    struct fi_info *hints = fi_allocinfo();
    if (!hints) {
        fprintf(stderr, "[Rank %d] fi_allocinfo failed\n", rank);
        return -1;
    }
    
    // NCCL-style configuration
    hints->caps = FI_MSG | FI_HMEM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT | FI_MR_HMEM |
                                  FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->av_type = FI_AV_UNSPEC;
    
    // Try different API versions
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &ctx->info);
    if (ret != 0) {
        ret = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0, hints, &ctx->info);
    }
    fi_freeinfo(hints);
    
    if (ret != 0) {
        fprintf(stderr, "[Rank %d] fi_getinfo failed: %s\n", rank, fi_strerror(-ret));
        return ret;
    }
    
    // Create fabric
    CHECK_FI(fi_fabric(ctx->info->fabric_attr, &ctx->fabric, NULL), "fi_fabric");
    
    // Create domain
    CHECK_FI(fi_domain(ctx->fabric, ctx->info, &ctx->domain, NULL), "fi_domain");
    
    // Create CQ
    struct fi_cq_attr cq_attr;
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.size = 128;
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    CHECK_FI(fi_cq_open(ctx->domain, &cq_attr, &ctx->cq, NULL), "fi_cq_open");
    
    // Create AV
    struct fi_av_attr av_attr;
    memset(&av_attr, 0, sizeof(av_attr));
    av_attr.type = ctx->info->domain_attr->av_type;
    CHECK_FI(fi_av_open(ctx->domain, &av_attr, &ctx->av, NULL), "fi_av_open");
    
    // Create endpoint
    CHECK_FI(fi_endpoint(ctx->domain, ctx->info, &ctx->ep, NULL), "fi_endpoint");
    
    // Bind CQ and AV to endpoint
    CHECK_FI(fi_ep_bind(ctx->ep, &ctx->cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind cq");
    CHECK_FI(fi_ep_bind(ctx->ep, &ctx->av->fid, 0), "fi_ep_bind av");
    
    // Enable endpoint
    CHECK_FI(fi_enable(ctx->ep), "fi_enable");
    
    return 0;
}

/*
 * Exchange addresses via MPI
 */
static int exchange_addresses(ofi_context_t *ctx, int peer_rank) {
    char my_addr[256];
    char peer_addr[256];
    size_t my_addr_len = sizeof(my_addr);
    
    // Get local address
    CHECK_FI(fi_getname(&ctx->ep->fid, my_addr, &my_addr_len), "fi_getname");
    
    // Exchange addresses via MPI
    size_t peer_addr_len = sizeof(peer_addr);
    MPI_Status status;
    
    if (rank < peer_rank) {
        MPI_Send(&my_addr_len, sizeof(my_addr_len), MPI_BYTE, peer_rank, 0, MPI_COMM_WORLD);
        MPI_Send(my_addr, my_addr_len, MPI_BYTE, peer_rank, 1, MPI_COMM_WORLD);
        MPI_Recv(&peer_addr_len, sizeof(peer_addr_len), MPI_BYTE, peer_rank, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(peer_addr, peer_addr_len, MPI_BYTE, peer_rank, 1, MPI_COMM_WORLD, &status);
    } else {
        MPI_Recv(&peer_addr_len, sizeof(peer_addr_len), MPI_BYTE, peer_rank, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(peer_addr, peer_addr_len, MPI_BYTE, peer_rank, 1, MPI_COMM_WORLD, &status);
        MPI_Send(&my_addr_len, sizeof(my_addr_len), MPI_BYTE, peer_rank, 0, MPI_COMM_WORLD);
        MPI_Send(my_addr, my_addr_len, MPI_BYTE, peer_rank, 1, MPI_COMM_WORLD);
    }
    
    // Insert peer address into AV
    int ret = fi_av_insert(ctx->av, peer_addr, 1, &ctx->peer_addr, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "[Rank %d] fi_av_insert failed: %s\n", rank, fi_strerror(-ret));
        return ret;
    }
    
    return 0;
}

/*
 * Test 1: Unidirectional bandwidth test with busy waiting
 * Rank 0 sends, Rank 1 receives
 */
static void test1_unidirectional_bw(ofi_context_t *ctx, struct fid_mr *mr, 
                                    void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL), 
                     "fi_send warmup");
            wait_for_completion(ctx->cq);
        } else {
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv warmup");
            wait_for_completion(ctx->cq);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test with busy waiting
    for (int i = 0; i < BANDWIDTH_ITERS; i++) {
        if (rank == 0) {
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send");
            wait_for_completion(ctx->cq);
        } else {
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv");
            wait_for_completion(ctx->cq);
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
 * Test 2: Ping-pong test with 1-byte ack
 * Rank 0 sends msg_size, Rank 1 receives and sends back 1 byte
 */
static void test2_pingpong_ack(ofi_context_t *ctx, struct fid_mr *mr,
                               void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send warmup");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, 1, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv warmup");
            wait_for_completion(ctx->cq);
        } else {
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv warmup");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_send(ctx->ep, gpu_buf, 1, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send warmup");
            wait_for_completion(ctx->cq);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test
    for (int i = 0; i < LATENCY_ITERS; i++) {
        if (rank == 0) {
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, 1, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv");
            wait_for_completion(ctx->cq);
        } else {
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_send(ctx->ep, gpu_buf, 1, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send");
            wait_for_completion(ctx->cq);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    if (rank == 0) {
        elapsed = end - start;
        // Round-trip time in microseconds
        double latency_us = (elapsed / LATENCY_ITERS) * 1000000.0;
        // Bandwidth: total data transferred (msg_size + 1 byte ack) per round-trip
        double total_data = (double)msg_size + 1.0;
        double bw_mbps = total_data / elapsed / LATENCY_ITERS / (1024.0 * 1024.0);
        printf("%-30zu%-30.6f%-30.6f\n", msg_size, latency_us, bw_mbps);
        fflush(stdout);
    }
}

/*
 * Test 3: Ping-pong test with same-size reply
 * Rank 0 sends msg_size, Rank 1 receives and sends back msg_size
 */
static void test3_pingpong_same(ofi_context_t *ctx, struct fid_mr *mr,
                                void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        if (rank == 0) {
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send warmup");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv warmup");
            wait_for_completion(ctx->cq);
        } else {
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv warmup");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send warmup");
            wait_for_completion(ctx->cq);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test
    for (int i = 0; i < LATENCY_ITERS; i++) {
        if (rank == 0) {
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv");
            wait_for_completion(ctx->cq);
        } else {
            CHECK_FI(fi_recv(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                     "fi_recv");
            wait_for_completion(ctx->cq);
            CHECK_FI(fi_send(ctx->ep, gpu_buf, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                     "fi_send");
            wait_for_completion(ctx->cq);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    if (rank == 0) {
        elapsed = end - start;
        // One-way latency in microseconds
        double latency_us = (elapsed / LATENCY_ITERS / 2.0) * 1000000.0;
        // Bandwidth: total data transferred (msg_size * 2) per round-trip
        double total_data = (double)msg_size * 2.0;
        double bw_mbps = total_data / elapsed / LATENCY_ITERS / (1024.0 * 1024.0);
        printf("%-30zu%-30.6f%-30.6f\n", msg_size, latency_us, bw_mbps);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    int size;
    ofi_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Set GPU
    CHECK_CUDA(cudaSetDevice(0));
    
    // Initialize independent libfabric instance
    if (init_ofi(&ctx) != 0) {
        MPI_Finalize();
        return 1;
    }
    
    // Exchange addresses
    int peer_rank = (rank == 0) ? 1 : 0;
    if (exchange_addresses(&ctx, peer_rank) != 0) {
        MPI_Finalize();
        return 1;
    }
    
    // Allocate GPU memory
    void *gpu_buf;
    CHECK_CUDA(cudaMalloc(&gpu_buf, MAX_MSG_SIZE));
    CHECK_CUDA(cudaMemset(gpu_buf, rank, MAX_MSG_SIZE));
    
    // Register GPU memory
    int gpu_id;
    CHECK_CUDA(cudaGetDevice(&gpu_id));
    
    struct iovec iov;
    iov.iov_base = gpu_buf;
    iov.iov_len = MAX_MSG_SIZE;
    
    struct fi_mr_attr mr_attr;
    memset(&mr_attr, 0, sizeof(mr_attr));
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE;
    mr_attr.iface = FI_HMEM_CUDA;
    mr_attr.device.cuda = gpu_id;
    
    struct fid_mr *mr;
    CHECK_FI(fi_mr_regattr(ctx.domain, &mr_attr, 0, &mr), "fi_mr_regattr");
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ========================================
    // Test 1: Unidirectional bandwidth test
    // ========================================
    if (rank == 0) {
        printf("=== Test 1: Unidirectional (Rank 0 -> Rank 1) ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test1_unidirectional_bw(&ctx, mr, gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ========================================
    // Test 2: Ping-pong with 1-byte ack
    // ========================================
    if (rank == 0) {
        printf("=== Test 2: Ping-Pong with 1-byte Ack ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test2_pingpong_ack(&ctx, mr, gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ========================================
    // Test 3: Ping-pong with same-size reply
    // ========================================
    if (rank == 0) {
        printf("=== Test 3: Ping-Pong with Same-size Reply ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test3_pingpong_same(&ctx, mr, gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n=== All Tests Complete ===\n");
    }
    
    // Cleanup
    fi_close(&mr->fid);
    cudaFree(gpu_buf);
    fi_close(&ctx.ep->fid);
    fi_close(&ctx.av->fid);
    fi_close(&ctx.cq->fid);
    fi_close(&ctx.domain->fid);
    fi_close(&ctx.fabric->fid);
    fi_freeinfo(ctx.info);
    
    MPI_Finalize();
    return 0;
}

