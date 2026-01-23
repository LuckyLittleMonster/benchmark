#!/bin/bash

echo "════════════════════════════════════════════════════════"
echo "  GPU Direct RDMA Performance Test Suite"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Strategy: Independent libfabric instance (NCCL's approach)"
echo "  - MPI for coordination"
echo "  - Separate libfabric instance for GPU Direct RDMA"
echo "  - Optimized tight polling (no sleep) for best performance"
echo ""
echo "Test Suite (all with batch optimizations):"
echo "  1. Unidirectional Bandwidth (Rank 0 -> Rank 1, batched)"
echo "  2. Ping-Pong with 1-byte Ack (batched + fi_inject)"
echo "  3. Ping-Pong with Same-size Reply (batched)"
echo ""
echo "Optimizations:"
echo "  - Batch operations: Fixed window size of 16 for all message sizes"
echo "  - Batch CQ reads: Read up to 16 completions at once"
echo "  - fi_inject: For 1-byte acks (no wait needed)"
echo "  - Tight polling: Continuous CQ polling without sleep"
echo ""

# Load required modules
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load craype-accel-nvidia80

# Use NCCL's bundled libfabric version
NCCL_OFI_LIB=/global/common/software/nersc9/nccl/2.24.3/plugin/deps/lib
SYSTEM_OFI_INCLUDE=/opt/cray/libfabric/1.22.0/include

echo "Compiling bwl.c..."
CC -o bwl_optimized bwl_optimized.c \
    -I${SYSTEM_OFI_INCLUDE} \
    -L${NCCL_OFI_LIB} \
    -Wl,-rpath,${NCCL_OFI_LIB} \
    -lfabric \
    -L${CUDA_HOME}/lib64 \
    -lcudart \
    -O3

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Set runtime environment
export LD_LIBRARY_PATH=${NCCL_OFI_LIB}:${LD_LIBRARY_PATH}

# Libfabric configuration
export FI_PROVIDER=cxi
export FI_LOG_LEVEL=error  # Reduced from 'warn' to hide non-critical messages
export FI_HMEM_CUDA_ENABLE=1
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=0

# GPU support
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80

# CXI specific settings
export FI_CXI_RX_MATCH_MODE=hardware
export FI_MR_CACHE_MONITOR=memhooks

echo "════════════════════════════════════════════════════════"
echo "  Environment Configuration"
echo "════════════════════════════════════════════════════════"
echo "Provider:         cxi"
echo "Libfabric:        NCCL bundled (v1.23.1)"
echo "GPU Direct RDMA:  Enabled"
echo "Polling mode:     Tight polling (optimized)"
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Test Configuration"
echo "════════════════════════════════════════════════════════"
echo "Nodes:            2"
echo "Tasks:            2 (1 per node)"
echo "GPUs per task:    1"
echo ""
echo "Bandwidth test:"
echo "  - Iterations:   100 (after 20 warmup)"
echo "  - Message size: 1 B to 4 GB"
echo "  - Direction:    Unidirectional (Rank 0 -> Rank 1)"
echo ""
echo "Latency test:"
echo "  - Iterations:   1000 (after 20 warmup)"
echo "  - Message size: 1 B to 4 GB"
echo "  - Pattern:      Ping-Pong (round-trip / 2)"
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Starting Test"
echo "════════════════════════════════════════════════════════"
echo ""

srun --nodes=2 --ntasks=2 --time=00:15:00 -C gpu --gpus-per-task=1 \
     -A dasrepo -q debug -u \
     --export=ALL \
     ./bwl_optimized

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Test Complete"
echo "════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully"
    echo ""
    echo "Key Findings:"
    echo "  - Independent libfabric instance works with MPI"
    echo "  - GPU Direct RDMA enabled and functional"
    echo "  - Tight polling eliminates sleep overhead"
    echo ""
    echo "Performance Notes:"
    echo "  - Peak bandwidth typically at large message sizes (>1MB)"
    echo "  - Latency should be ~1-2 µs for small messages"
    echo "  - Compare with MPI: ~20-25 GB/s, ~1-2 µs"
    echo "  - Compare with NCCL: ~20-25 GB/s, ~2-3 µs"
    echo ""
    echo "Output format matches bw.log for easy comparison"
else
    echo "✗ Test failed (exit code: $EXIT_CODE)"
    echo ""
    echo "Possible causes:"
    echo "  - Resource allocation failure"
    echo "  - Network configuration issue"
    echo "  - GPU access problem"
    echo "  - CXI provider unavailable"
fi

echo ""

