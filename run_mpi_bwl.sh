#!/bin/bash

echo "════════════════════════════════════════════════════════"
echo "  MPI GPU Direct RDMA Performance Test"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Communication: MPI Send/Recv (GPU-aware)"
echo "Test Suite:"
echo "  1. Unidirectional Bandwidth (Rank 0 -> Rank 1)"
echo "  2. Ping-Pong with 1-byte Ack"
echo "  3. Ping-Pong with Same-size Reply"
echo ""

# Load modules
module load PrgEnv-gnu
module load cudatoolkit/12.4
module load craype-accel-nvidia80

echo "Compiling mpi_bwl.c..."
CC -o mpi_bwl mpi_bwl.c \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -lcudart \
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

echo "════════════════════════════════════════════════════════"
echo "  Environment Configuration"
echo "════════════════════════════════════════════════════════"
echo "GPU Support:     Enabled (Cray MPICH)"
echo "GPU Direct RDMA: Enabled"
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Starting Test (2 nodes, 1 task per node)"
echo "════════════════════════════════════════════════════════"
echo ""

srun --nodes=2 --ntasks-per-node=1 --time=00:15:00 -C gpu --gpus-per-task=1 \
     -A dasrepo -q debug -u \
     --export=ALL \
     ./mpi_bwl

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
    echo "  - MPI uses libfabric (CXI provider) internally"
    echo "  - GPU Direct RDMA enabled automatically"
    echo "  - Compare with bwl.c (direct libfabric) and nccl_bwl.c (NCCL)"
else
    echo ""
    echo "✗ Test failed (exit code: $EXIT_CODE)"
fi

