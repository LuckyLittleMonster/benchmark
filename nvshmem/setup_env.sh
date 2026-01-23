#!/bin/bash
# NVSHMEM Environment Setup Script
# Source this file to configure your environment for NVSHMEM

echo "=== NVSHMEM Environment Setup ==="

# 1. NVSHMEM Home (modify according to your installation)
if [ -z "$NVSHMEM_HOME" ]; then
    # Try to detect common installation paths
    NVSHMEM_CANDIDATES=(
        "/opt/nvshmem"
        "/usr/local/nvshmem"
        "$HOME/nvshmem"
        "/sw/summit/nvshmem"
        "/opt/nvidia/hpc_sdk/Linux_x86_64/*/comm_libs/nvshmem"
    )
    
    for candidate in "${NVSHMEM_CANDIDATES[@]}"; do
        if [ -d "$candidate" ]; then
            export NVSHMEM_HOME="$candidate"
            echo "Found NVSHMEM at: $NVSHMEM_HOME"
            break
        fi
    done
    
    if [ -z "$NVSHMEM_HOME" ]; then
        echo "Warning: NVSHMEM_HOME not found. Please set it manually:"
        echo "  export NVSHMEM_HOME=/path/to/nvshmem"
    fi
else
    echo "NVSHMEM_HOME: $NVSHMEM_HOME"
fi

# 2. CUDA Home
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &> /dev/null; then
        export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
    fi
fi
echo "CUDA_HOME: $CUDA_HOME"

# 3. MPI Home (optional, for MPI bootstrap)
if [ -z "$MPI_HOME" ] && command -v mpicc &> /dev/null; then
    export MPI_HOME="$(dirname $(dirname $(which mpicc)))"
    echo "MPI_HOME: $MPI_HOME"
fi

# 4. Update PATH and LD_LIBRARY_PATH
if [ -n "$NVSHMEM_HOME" ]; then
    export PATH="$NVSHMEM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# 5. NVSHMEM Configuration
# Bootstrap method: MPI or PMI
export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}
echo "NVSHMEM_BOOTSTRAP: $NVSHMEM_BOOTSTRAP"

# Transport layer
# Options: ibrc (InfiniBand RC), ibud (InfiniBand UD), tcp
if lspci | grep -i mellanox &> /dev/null || lspci | grep -i infiniband &> /dev/null; then
    export NVSHMEM_TRANSPORT=${NVSHMEM_TRANSPORT:-ibrc}
    echo "Detected InfiniBand, using transport: $NVSHMEM_TRANSPORT"
else
    export NVSHMEM_TRANSPORT=${NVSHMEM_TRANSPORT:-tcp}
    echo "Using transport: $NVSHMEM_TRANSPORT"
fi

# 6. GPU Direct RDMA Configuration
# Disable CUDA VMM for better GPUDirect RDMA compatibility
export NVSHMEM_DISABLE_CUDA_VMM=1

# 7. InfiniBand specific settings (if applicable)
if [ "$NVSHMEM_TRANSPORT" = "ibrc" ] || [ "$NVSHMEM_TRANSPORT" = "ibud" ]; then
    # GID index (use 3 for RoCE v2, 0 for IB)
    export NVSHMEM_IB_GID_INDEX=${NVSHMEM_IB_GID_INDEX:-3}
    
    # Traffic class for QoS
    export NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-105}
    
    echo "InfiniBand GID Index: $NVSHMEM_IB_GID_INDEX"
    echo "InfiniBand Traffic Class: $NVSHMEM_IB_TRAFFIC_CLASS"
fi

# 8. Debug settings (uncomment for debugging)
# export NVSHMEM_DEBUG=TRACE
# export NVSHMEM_DEBUG_SUBSYS=ALL

# 9. Performance tuning
# Symmetric heap size (default is usually sufficient)
# export NVSHMEM_SYMMETRIC_SIZE=4G

echo ""
echo "=== Environment Ready ==="
echo "You can now compile and run NVSHMEM programs:"
echo "  make"
echo "  ./run_test.sh"
echo ""
echo "To verify NVSHMEM installation:"
if [ -n "$NVSHMEM_HOME" ] && [ -f "$NVSHMEM_HOME/examples/hello_world.c" ]; then
    echo "  cd $NVSHMEM_HOME/examples && make && nvshmrun -np 2 ./hello_world"
fi
echo ""

# 10. System information
echo "=== System Information ==="
echo "CUDA Devices:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n 2
fi

echo ""
echo "Network Devices:"
if command -v ibv_devices &> /dev/null; then
    ibv_devices
elif command -v ip &> /dev/null; then
    ip link show | grep -E "^[0-9]+:" | grep -v lo
fi

echo ""
echo "=== Setup Complete ==="

