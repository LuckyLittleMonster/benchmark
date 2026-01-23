#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <nvshmem.h>
#include <nvshmemx.h>

using std::cout;
using std::endl;
using std::string;

string rjust(string s, int width) {
	return string(width - s.size(), ' ') + s;
}

string ljust(string s, int width) {
	return s + string(width - s.size(), ' ');
}

#define check_cuda_error(func) do { \
	func; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// NVSHMEM kernel: Put data from local GPU to remote GPU
__global__ void nvshmem_put_kernel(float* dst, float* src, size_t num_floats, int target_pe) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	
	// Simple put operation
	for (size_t i = idx; i < num_floats; i += stride) {
		nvshmem_float_p(dst + i, src[i], target_pe);
	}
}

// NVSHMEM kernel: Get data from remote GPU to local GPU
__global__ void nvshmem_get_kernel(float* dst, float* src, size_t num_floats, int source_pe) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	
	// Simple get operation
	for (size_t i = idx; i < num_floats; i += stride) {
		dst[i] = nvshmem_float_g(src + i, source_pe);
	}
}

// NVSHMEM kernel: Direct memory access (using symmetric heap)
__global__ void nvshmem_direct_access_kernel(float* dst, float* src, size_t num_floats) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	
	for (size_t i = idx; i < num_floats; i += stride) {
		dst[i] = src[i];
	}
}

void test_put_bandwidth(float* src_ptr, float* dst_ptr, int target_pe, 
                        size_t min_size, size_t max_size, int iterations, int skip) {
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	
	cout << "\n=== NVSHMEM PUT Bandwidth (Local PE " << nvshmem_my_pe() 
	     << " -> Remote PE " << target_pe << ") ===" << endl;
	cout << ljust("size", 20) << "\t" << ljust("time(ms)", 20) << "\t" 
	     << ljust("bw(GB/s)", 20) << endl;
	
	for (size_t size = min_size; size <= max_size; size *= 2) {
		float total_time = 0.0;
		size_t num_floats = size / sizeof(float);
		
		int blockSize = 256;
		int numBlocks = (num_floats + blockSize - 1) / blockSize;
		numBlocks = (numBlocks > 1024) ? 1024 : numBlocks;
		
		// Warm-up iterations
		for (int i = 0; i < skip; i++) {
			nvshmem_put_kernel<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, target_pe);
			nvshmemx_quiet_on_stream(0);
		}
		cudaDeviceSynchronize();
		
		// Timed iterations
		cudaEventRecord(start_event, 0);
		for (int i = 0; i < (iterations - skip); i++) {
			nvshmem_put_kernel<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, target_pe);
			nvshmemx_quiet_on_stream(0);
		}
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&total_time, start_event, stop_event);
		
		double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
		cout << ljust(std::to_string(size), 20) << "\t" 
		     << ljust(std::to_string(total_time), 20) << "\t" 
		     << ljust(std::to_string(bandwidth), 20) << endl;
	}
	
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
}

void test_get_bandwidth(float* src_ptr, float* dst_ptr, int source_pe, 
                        size_t min_size, size_t max_size, int iterations, int skip) {
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	
	cout << "\n=== NVSHMEM GET Bandwidth (Remote PE " << source_pe 
	     << " -> Local PE " << nvshmem_my_pe() << ") ===" << endl;
	cout << ljust("size", 20) << "\t" << ljust("time(ms)", 20) << "\t" 
	     << ljust("bw(GB/s)", 20) << endl;
	
	for (size_t size = min_size; size <= max_size; size *= 2) {
		float total_time = 0.0;
		size_t num_floats = size / sizeof(float);
		
		int blockSize = 256;
		int numBlocks = (num_floats + blockSize - 1) / blockSize;
		numBlocks = (numBlocks > 1024) ? 1024 : numBlocks;
		
		// Warm-up iterations
		for (int i = 0; i < skip; i++) {
			nvshmem_get_kernel<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, source_pe);
			nvshmemx_quiet_on_stream(0);
		}
		cudaDeviceSynchronize();
		
		// Timed iterations
		cudaEventRecord(start_event, 0);
		for (int i = 0; i < (iterations - skip); i++) {
			nvshmem_get_kernel<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, source_pe);
			nvshmemx_quiet_on_stream(0);
		}
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&total_time, start_event, stop_event);
		
		double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
		cout << ljust(std::to_string(size), 20) << "\t" 
		     << ljust(std::to_string(total_time), 20) << "\t" 
		     << ljust(std::to_string(bandwidth), 20) << endl;
	}
	
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
}

int main(int argc, char *argv[]) {
	// Don't set bootstrap - let NVSHMEM auto-detect or use default
	// This should work with srun without needing MPI/PMI explicitly
	
	// Initialize NVSHMEM
	nvshmem_init();
	
	int my_pe = nvshmem_my_pe();
	int n_pes = nvshmem_n_pes();
	
	if (n_pes < 2) {
		if (my_pe == 0) {
			cout << "Error: This test requires at least 2 PEs (processes)" << endl;
			cout << "Usage: srun -n <num_processes> ./nvshmem_bw" << endl;
			cout << "Example: srun -n 2 ./nvshmem_bw" << endl;
			cout << "Or for specific GPUs: srun -n 2 --gpus-per-task=1 ./nvshmem_bw" << endl;
		}
		nvshmem_finalize();
		return 1;
	}
	
	// Test parameters
	size_t min_size = 1024;  // 1KB
	size_t max_size = 256ul * 1024 * 1024;  // 256MB
	int iterations = 32;
	int skip = 3;
	
	// Allocate symmetric memory (accessible by all PEs)
	float* src_ptr = (float*)nvshmem_malloc(max_size);
	float* dst_ptr = (float*)nvshmem_malloc(max_size);
	
	if (src_ptr == nullptr || dst_ptr == nullptr) {
		fprintf(stderr, "PE %d: Failed to allocate symmetric memory\n", my_pe);
		nvshmem_finalize();
		return 1;
	}
	
	// Initialize data on GPU
	check_cuda_error(cudaMemset(src_ptr, 1, max_size));
	check_cuda_error(cudaMemset(dst_ptr, 0, max_size));
	
	// Synchronize all PEs
	nvshmem_barrier_all();
	
	if (my_pe == 0) {
		cout << "=== NVSHMEM GPU Direct RDMA Bandwidth Test ===" << endl;
		cout << "Number of PEs: " << n_pes << endl;
		cout << "Testing PE 0 <-> PE 1" << endl;
		
		// PE 0 tests PUT to PE 1 and GET from PE 1
		test_put_bandwidth(src_ptr, dst_ptr, 1, min_size, max_size, iterations, skip);
		nvshmem_barrier_all();
		
		test_get_bandwidth(src_ptr, dst_ptr, 1, min_size, max_size, iterations, skip);
		nvshmem_barrier_all();
		
	} else if (my_pe == 1) {
		// PE 1 waits for PE 0's PUT test
		nvshmem_barrier_all();
		
		// PE 1 waits for PE 0's GET test
		nvshmem_barrier_all();
	} else {
		// Other PEs just wait
		nvshmem_barrier_all();
		nvshmem_barrier_all();
	}
	
	// Cleanup
	nvshmem_free(src_ptr);
	nvshmem_free(dst_ptr);
	nvshmem_finalize();
	
	return 0;
}

