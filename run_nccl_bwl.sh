#!/bin/bash

echo "════════════════════════════════════════════════════════"
echo "  NCCL GPU Direct RDMA Performance Test"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Communication: NCCL Send/Recv (Point-to-Point)"
echo "Test Suite:"
echo "  1. Unidirectional Bandwidth (Rank 0 -> Rank 1)"
echo "  2. Ping-Pong with 1-byte Ack"
echo "  3. Ping-Pong with Same-size Reply"
echo ""

# Load modules
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load nccl

# NCCL paths
NCCL_HOME=/global/common/software/nersc9/nccl/2.24.3

echo "Compiling nccl_bwl.c..."
CC -o nccl_bwl nccl_bwl.c \
    -I${CUDA_HOME}/include \
    -I${NCCL_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -L${NCCL_HOME}/lib \
    -Wl,-rpath,${NCCL_HOME}/lib \
    -lcudart \
    -lnccl \
    -O3

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Environment
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${LD_LIBRARY_PATH}

# NCCL environment variables
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn

echo "════════════════════════════════════════════════════════"
echo "  Environment Configuration"
echo "════════════════════════════════════════════════════════"
echo "NCCL Version:    2.24.3"
echo "NCCL Plugin:     aws-ofi-nccl (libfabric + CXI)"
echo "GPU Direct RDMA: Enabled (GDR Level: PHB)"
echo "Network:         Slingshot 11 (CXI)"
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Starting Test (2 nodes, 1 task per node)"
echo "════════════════════════════════════════════════════════"
echo ""

srun --nodes=2 --ntasks-per-node=1 --time=00:15:00 -C gpu --gpus-per-task=1 \
     -A dasrepo -q debug -u \
     --export=ALL \
     ./nccl_bwl

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Test Complete"
echo "════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Test completed successfully"
    echo ""
    echo "Performance Notes:"
    echo "  - NCCL uses aws-ofi-nccl plugin (libfabric + CXI)"
    echo "  - GPU Direct RDMA enabled automatically"
    echo "  - Compare with bwl.c (direct libfabric) and mpi_bwl.c (MPI)"
    echo ""
    echo "NCCL typically has:"
    echo "  - Slightly higher latency than direct libfabric (more layers)"
    echo "  - Similar bandwidth to MPI and libfabric"
    echo "  - Better collective operations (not tested here)"
else
    echo ""
    echo "✗ Test failed (exit code: $EXIT_CODE)"
fi

