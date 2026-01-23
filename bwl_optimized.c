/*
 * Optimized GPU Direct RDMA Bandwidth & Latency Test
 * Demonstrates various optimization techniques
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <cuda_runtime.h>
#include <sys/uio.h>

#define MAX_MSG_SIZE (10UL * 1024 * 1024)  // 10 MB max message size for testing
#define BUFFER_SIZE (1UL * 1024 * 1024 * 1024)  // 1 GB buffer for batch operations
#define WARMUP_ITERS 3
#define BANDWIDTH_ITERS 29
#define LATENCY_ITERS 29
#define WINDOW_SIZE 16  // Pipeline window size for optimization
#define INJECT_THRESHOLD 64  // Use fi_inject for messages <= 64 bytes

// Global variables
static int rank;

// OFI context
typedef struct {
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_ep *ep;
    struct fid_cq *cq;
    struct fid_av *av;
    struct fi_info *info;
    fi_addr_t peer_addr;
} ofi_context_t;

#define CHECK_FI(call, msg) do { \
    int ret = (call); \
    if (ret < 0) { \
        fprintf(stderr, "[Rank %d] %s: %s (%d)\n", rank, msg, fi_strerror(-ret), ret); \
        MPI_Abort(MPI_COMM_WORLD, ret); \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA Error: %s\n", rank, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

/*
 * Wait for N CQ completions with tight polling
 */
static int wait_for_n_completions(struct fid_cq *cq, int count) {
    struct fi_cq_entry entries[WINDOW_SIZE];
    int ret;
    int completed = 0;
    
    while (completed < count) {
        ret = fi_cq_read(cq, entries, WINDOW_SIZE);
        if (ret > 0) {
            completed += ret;
        } else if (ret < 0 && ret != -FI_EAGAIN) {
            struct fi_cq_err_entry err_entry;
            fi_cq_readerr(cq, &err_entry, 0);
            fprintf(stderr, "[Rank %d] CQ error: %s\n", rank, fi_strerror(err_entry.err));
            return ret;
        }
        // No timeout - keep polling until done
    }
    
    return 0;
}

/*
 * Single completion wait (for compatibility)
 */
static int wait_for_completion(struct fid_cq *cq) {
    return wait_for_n_completions(cq, 1);
}

/*
 * Initialize libfabric context (independent instance after MPI)
 */
static int init_fabric(ofi_context_t *ctx) {
    struct fi_info *hints = fi_allocinfo();
    if (!hints) {
        fprintf(stderr, "[Rank %d] Failed to allocate hints\n", rank);
        return -1;
    }
    
    // Configure hints matching NCCL's approach
    hints->caps = FI_LOCAL_COMM | FI_REMOTE_COMM | FI_MSG | FI_HMEM;
    hints->mode = FI_CONTEXT | FI_CONTEXT2;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->av_type = FI_AV_UNSPEC;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT | FI_MR_HMEM |
                                  FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    
    // Get provider info
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &ctx->info);
    if (ret != 0) {
        ret = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0, hints, &ctx->info);
        if (ret != 0) {
            fprintf(stderr, "[Rank %d] fi_getinfo failed: %s\n", rank, fi_strerror(-ret));
            fi_freeinfo(hints);
            return ret;
        }
    }
    fi_freeinfo(hints);
    
    // Create fabric
    CHECK_FI(fi_fabric(ctx->info->fabric_attr, &ctx->fabric, NULL), "fi_fabric");
    
    // Create domain
    CHECK_FI(fi_domain(ctx->fabric, ctx->info, &ctx->domain, NULL), "fi_domain");
    
    // Create completion queue
    struct fi_cq_attr cq_attr;
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;  // Busy polling
    cq_attr.size = 256;  // Larger queue for pipelining
    CHECK_FI(fi_cq_open(ctx->domain, &cq_attr, &ctx->cq, NULL), "fi_cq_open");
    
    // Create endpoint
    CHECK_FI(fi_endpoint(ctx->domain, ctx->info, &ctx->ep, NULL), "fi_endpoint");
    
    // Create address vector
    struct fi_av_attr av_attr;
    memset(&av_attr, 0, sizeof(av_attr));
    av_attr.type = ctx->info->domain_attr->av_type;
    av_attr.count = 2;
    CHECK_FI(fi_av_open(ctx->domain, &av_attr, &ctx->av, NULL), "fi_av_open");
    
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
    
    CHECK_FI(fi_getname(&ctx->ep->fid, my_addr, &my_addr_len), "fi_getname");
    
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
    
    int ret = fi_av_insert(ctx->av, peer_addr, 1, &ctx->peer_addr, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "[Rank %d] fi_av_insert failed: %s\n", rank, fi_strerror(-ret));
        return ret;
    }
    
    return 0;
}

/*
 * OPTIMIZED Test 1: Unidirectional bandwidth with pipelining
 * Key optimization: Pre-post multiple receives, batch sends
 */
static void test1_optimized_unidirectional(ofi_context_t *ctx, struct fid_mr *mr, 
                                          void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Use fixed window size for all messages
    int window = WINDOW_SIZE;
    
    // Simple warmup (no pipelining to avoid complexity)
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
    
    // Actual test with pipelining
    int remaining = BANDWIDTH_ITERS;
    while (remaining > 0) {
        int batch = (remaining > window) ? window : remaining;
        
        if (rank == 0) {
            // Batch sends using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_send(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                         "fi_send");
            }
            wait_for_n_completions(ctx->cq, batch);
        } else {
            // Batch receives using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_recv(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                         "fi_recv");
            }
            wait_for_n_completions(ctx->cq, batch);
        }
        
        remaining -= batch;
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
 * OPTIMIZED Test 2: Ping-pong with fi_inject and batching
 * Key optimizations: 
 *   1. Use fi_inject for 1-byte ack (no wait needed)
 *   2. Batch operations to reduce synchronization overhead
 */
static void test2_optimized_pingpong_ack(ofi_context_t *ctx, struct fid_mr *mr,
                                        void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Use fixed window size for all messages
    int window = WINDOW_SIZE;
    
    // Simple warmup
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
            int ret = fi_inject(ctx->ep, gpu_buf, 1, ctx->peer_addr);
            if (ret < 0) {
                CHECK_FI(fi_send(ctx->ep, gpu_buf, 1, fi_mr_desc(mr), ctx->peer_addr, NULL),
                         "fi_send warmup");
                wait_for_completion(ctx->cq);
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Actual test with batching
    int remaining = LATENCY_ITERS;
    while (remaining > 0) {
        int batch = (remaining > window) ? window : remaining;
        
        if (rank == 0) {
            // Batch: send all messages using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_send(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                         "fi_send");
            }
            wait_for_n_completions(ctx->cq, batch);
            
            // Batch: receive all acks using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_recv(ctx->ep, buf_offset, 1, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                         "fi_recv");
            }
            wait_for_n_completions(ctx->cq, batch);
        } else {
            // Batch: receive all messages using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_recv(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                         "fi_recv");
            }
            wait_for_n_completions(ctx->cq, batch);
            
            // Batch: send all acks using fi_inject with different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                int ret = fi_inject(ctx->ep, buf_offset, 1, ctx->peer_addr);
                if (ret < 0) {
                    CHECK_FI(fi_send(ctx->ep, buf_offset, 1, fi_mr_desc(mr), ctx->peer_addr, NULL),
                             "fi_send");
                    wait_for_completion(ctx->cq);
                }
            }
        }
        
        remaining -= batch;
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
 * OPTIMIZED Test 3: Ping-pong with same-size reply and batching
 * Key optimization: Batch operations to reduce synchronization overhead
 */
static void test3_optimized_pingpong_same(ofi_context_t *ctx, struct fid_mr *mr,
                                         void *gpu_buf, size_t msg_size) {
    double start, end, elapsed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Use fixed window size for all messages
    int window = WINDOW_SIZE;
    
    // Simple warmup
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
    
    // Actual test with batching
    int remaining = LATENCY_ITERS;
    while (remaining > 0) {
        int batch = (remaining > window) ? window : remaining;
        
        if (rank == 0) {
            // Batch: send all messages using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_send(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                         "fi_send");
            }
            wait_for_n_completions(ctx->cq, batch);
            
            // Batch: receive all replies using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_recv(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                         "fi_recv");
            }
            wait_for_n_completions(ctx->cq, batch);
        } else {
            // Batch: receive all messages using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_recv(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), FI_ADDR_UNSPEC, NULL),
                         "fi_recv");
            }
            wait_for_n_completions(ctx->cq, batch);
            
            // Batch: send all replies using different buffer offsets
            for (int i = 0; i < batch; i++) {
                char *buf_offset = (char *)gpu_buf + (i * msg_size);
                CHECK_FI(fi_send(ctx->ep, buf_offset, msg_size, fi_mr_desc(mr), ctx->peer_addr, NULL),
                         "fi_send");
            }
            wait_for_n_completions(ctx->cq, batch);
        }
        
        remaining -= batch;
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
    int size;
    ofi_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    
    // Initialize MPI first
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
        printf("=== Optimized GPU Direct RDMA Performance Tests ===\n");
        printf("Configuration:\n");
        printf("  - Max message size: %lu MB\n", MAX_MSG_SIZE / (1024 * 1024));
        printf("  - GPU buffer size: %lu MB (for batch operations)\n", BUFFER_SIZE / (1024 * 1024));
        printf("  - Batch window: %d operations\n\n", WINDOW_SIZE);
        printf("Optimizations applied to all tests:\n");
        printf("  1. Batch operations: Pre-post up to %d operations\n", WINDOW_SIZE);
        printf("  2. Buffer offsets: Each operation uses different offset\n");
        printf("  3. Batch CQ read: Read up to %d completions at once\n", WINDOW_SIZE);
        printf("  4. fi_inject: For 1-byte ack in Test 2 (no wait needed)\n");
        printf("  5. Tight polling: No sleep, continuous CQ polling\n\n");
    }
    
    // Initialize libfabric (independent instance after MPI)
    if (init_fabric(&ctx) != 0) {
        MPI_Finalize();
        return -1;
    }
    
    // Exchange addresses
    int peer_rank = (rank == 0) ? 1 : 0;
    if (exchange_addresses(&ctx, peer_rank) != 0) {
        MPI_Finalize();
        return -1;
    }
    
    // Allocate GPU memory (1 GB for batch operations)
    void *gpu_buf;
    size_t buf_size = BUFFER_SIZE;  // Use BUFFER_SIZE (1 GB) not MAX_MSG_SIZE (10 MB)
    CHECK_CUDA(cudaMalloc(&gpu_buf, buf_size));
    CHECK_CUDA(cudaMemset(gpu_buf, 0, buf_size));
    
    // Register GPU memory
    struct iovec iov;
    iov.iov_base = gpu_buf;
    iov.iov_len = buf_size;
    
    struct fi_mr_attr mr_attr;
    memset(&mr_attr, 0, sizeof(mr_attr));
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    mr_attr.offset = 0;
    mr_attr.requested_key = 0;
    mr_attr.context = NULL;
    mr_attr.iface = FI_HMEM_CUDA;
    mr_attr.device.cuda = gpu_id;
    
    struct fid_mr *mr;
    CHECK_FI(fi_mr_regattr(ctx.domain, &mr_attr, 0, &mr), "fi_mr_regattr");
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 1: Optimized Unidirectional
    if (rank == 0) {
        printf("=== Test 1: Unidirectional Bandwidth (Rank 0 -> Rank 1, batched) ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test1_optimized_unidirectional(&ctx, mr, gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 2: Optimized Ping-pong with inject
    if (rank == 0) {
        printf("=== Test 2: Ping-Pong with 1-byte Ack (batched + fi_inject) ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test2_optimized_pingpong_ack(&ctx, mr, gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 3: Ping-pong same-size
    if (rank == 0) {
        printf("=== Test 3: Ping-Pong Same-size Reply (batched) ===\n");
        printf("%-30s%-30s%-30s\n", "size", "latency(us)", "BW(MB/s)");
    }
    
    for (size_t msg_size = 1; msg_size <= MAX_MSG_SIZE; msg_size *= 2) {
        test3_optimized_pingpong_same(&ctx, mr, gpu_buf, msg_size);
    }
    
    if (rank == 0) {
        printf("\n=== All Tests Complete ===\n");
    }
    
    // Cleanup
    fi_close(&mr->fid);
    cudaFree(gpu_buf);
    fi_close(&ctx.ep->fid);
    fi_close(&ctx.cq->fid);
    fi_close(&ctx.av->fid);
    fi_close(&ctx.domain->fid);
    fi_close(&ctx.fabric->fid);
    fi_freeinfo(ctx.info);
    
    MPI_Finalize();
    return 0;
}

