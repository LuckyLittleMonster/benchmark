#!/bin/bash

echo "════════════════════════════════════════════════════════"
echo "  Device-Side NCCL Intra-Node Bandwidth Test"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Communication: Device-side NCCL Send/Recv (Point-to-Point)"
echo "Test: Unidirectional Bandwidth (GPU0 -> GPU1)"
echo ""

# Load modules for Delta
module load PrgEnv-gnu
module load cudatoolkit/25.3_12.8
module load craype-accel-nvidia80
# module load nccl  # Not available on this system

# NCCL paths (adjust for NCCL 2.28.9 if needed)
NCCL_HOME=${NCCL_HOME:-/u/$USER/software/nccl-2.28}
CUDA_HOME=${CUDA_HOME:-${CUDATOOLKIT_HOME}}

echo "Using NCCL from: $NCCL_HOME"
echo "Using CUDA from: $CUDA_HOME"
echo ""

# Compile
echo "Compiling intra_dev_nccl_bwl.cu..."
nvcc -o intra_dev_nccl_bwl intra_dev_nccl_bwl.cu \
    -I${CUDA_HOME}/include \
    -I${NCCL_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -L${NCCL_HOME}/lib \
    -Xlinker -rpath,${NCCL_HOME}/lib \
    -lcudart \
    -lnccl \
    -O3 \
    -std=c++17

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Set environment
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${LD_LIBRARY_PATH}
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn

# Run
echo "Running intra_dev_nccl_bwl..."
echo ""
./intra_dev_nccl_bwl

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Execution failed!"
    exit 1
fi

echo ""
echo "✓ Test completed successfully"
