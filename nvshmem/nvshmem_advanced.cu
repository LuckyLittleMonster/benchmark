#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <nvshmem.h>
#include <nvshmemx.h>

using std::cout;
using std::endl;
using std::string;

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

// Advanced kernel: Direct pointer access on symmetric heap
// This demonstrates accessing remote GPU memory directly like local memory
__global__ void nvshmem_direct_memcpy_kernel(float* dst, const float* src, 
                                              size_t num_floats, int source_pe) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	
	// Direct memory access - NVSHMEM allows treating remote memory as local
	// Note: This requires the pointer to be from symmetric heap
	for (size_t i = idx; i < num_floats; i += stride) {
		// Using nvshmem_float_g for single element get
		dst[i] = nvshmem_float_g(src + i, source_pe);
	}
}

// Kernel using atomic operations on remote memory
__global__ void nvshmem_atomic_kernel(float* remote_counter, int target_pe, int iterations) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx == 0) {
		for (int i = 0; i < iterations; i++) {
			// Atomic add on remote GPU memory
			nvshmem_float_atomic_add(remote_counter, 1.0f, target_pe);
		}
	}
}

// Kernel using collective operations
__global__ void nvshmem_broadcast_kernel(float* data, size_t num_floats, int root_pe) {
	// Broadcast data from root PE to all PEs
	nvshmem_float_broadcast(NVSHMEM_TEAM_WORLD, data, data, num_floats, root_pe);
}

// Kernel using signal/wait for fine-grained synchronization
__global__ void nvshmem_signal_kernel(uint64_t* signal_ptr, int target_pe, uint64_t value) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		nvshmem_uint64_signal(signal_ptr, value, target_pe);
	}
}

__global__ void nvshmem_wait_kernel(uint64_t* signal_ptr, uint64_t value) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		nvshmem_uint64_wait_until(signal_ptr, NVSHMEM_CMP_GE, value);
	}
}

// Test 1: Direct memory access (pointer-to-pointer)
void test_direct_access(float* src_ptr, float* dst_ptr, int remote_pe, size_t size) {
	int my_pe = nvshmem_my_pe();
	cout << "\n=== Test 1: Direct Memory Access (PE " << my_pe << " -> PE " << remote_pe << ") ===" << endl;
	
	size_t num_floats = size / sizeof(float);
	int blockSize = 256;
	int numBlocks = (num_floats + blockSize - 1) / blockSize;
	numBlocks = (numBlocks > 1024) ? 1024 : numBlocks;
	
	// Initialize data
	check_cuda_error(cudaMemset(src_ptr, 0, size));
	float init_value = (float)my_pe + 100.0f;
	check_cuda_error(cudaMemset(dst_ptr, *(int*)&init_value, size));
	
	nvshmem_barrier_all();
	
	// Perform direct memory access
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	nvshmem_direct_memcpy_kernel<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, remote_pe);
	nvshmemx_quiet_on_stream(0);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float elapsed_ms;
	cudaEventElapsedTime(&elapsed_ms, start, stop);
	double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
	
	cout << "Size: " << size / (1024 * 1024) << " MB, "
	     << "Time: " << elapsed_ms << " ms, "
	     << "Bandwidth: " << bandwidth << " GB/s" << endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Test 2: Atomic operations
void test_atomic_operations(float* counter_ptr, int remote_pe) {
	int my_pe = nvshmem_my_pe();
	cout << "\n=== Test 2: Atomic Operations (PE " << my_pe << " -> PE " << remote_pe << ") ===" << endl;
	
	// Initialize counter
	check_cuda_error(cudaMemset(counter_ptr, 0, sizeof(float)));
	nvshmem_barrier_all();
	
	int iterations = 1000;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	nvshmem_atomic_kernel<<<1, 1>>>(counter_ptr, remote_pe, iterations);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float elapsed_ms;
	cudaEventElapsedTime(&elapsed_ms, start, stop);
	
	nvshmem_barrier_all();
	
	// Verify on remote PE
	if (my_pe == remote_pe) {
		float counter_value;
		cudaMemcpy(&counter_value, counter_ptr, sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Counter value on PE " << remote_pe << ": " << counter_value 
		     << " (expected: " << iterations << ")" << endl;
	}
	
	cout << "Atomic operations: " << iterations << " ops, "
	     << "Time: " << elapsed_ms << " ms, "
	     << "Latency: " << (elapsed_ms * 1000.0 / iterations) << " us/op" << endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Test 3: Signal/Wait for synchronization
void test_signal_wait(uint64_t* signal_ptr) {
	int my_pe = nvshmem_my_pe();
	int n_pes = nvshmem_n_pes();
	
	cout << "\n=== Test 3: Signal/Wait Synchronization (PE " << my_pe << ") ===" << endl;
	
	// Initialize signal
	check_cuda_error(cudaMemset(signal_ptr, 0, sizeof(uint64_t)));
	nvshmem_barrier_all();
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	if (my_pe == 0) {
		// PE 0 waits for signal from PE 1
		cout << "PE 0: Waiting for signal from PE 1..." << endl;
		cudaEventRecord(start);
		nvshmem_wait_kernel<<<1, 1>>>(signal_ptr, 42);
		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		
		float elapsed_ms;
		cudaEventElapsedTime(&elapsed_ms, start, stop);
		cout << "PE 0: Received signal! Wait time: " << elapsed_ms << " ms" << endl;
		
	} else if (my_pe == 1) {
		// PE 1 sends signal to PE 0 after a delay
		cout << "PE 1: Sending signal to PE 0..." << endl;
		cudaDeviceSynchronize();
		// Small delay to demonstrate wait
		for (int i = 0; i < 1000000; i++) { /* busy wait */ }
		nvshmem_signal_kernel<<<1, 1>>>(signal_ptr, 0, 42);
		cudaDeviceSynchronize();
		cout << "PE 1: Signal sent!" << endl;
	}
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	nvshmem_barrier_all();
}

// Test 4: Broadcast operation
void test_broadcast(float* data_ptr, size_t size) {
	int my_pe = nvshmem_my_pe();
	int n_pes = nvshmem_n_pes();
	int root_pe = 0;
	
	cout << "\n=== Test 4: Broadcast Operation (Root PE: " << root_pe << ") ===" << endl;
	
	size_t num_floats = size / sizeof(float);
	
	// Initialize data on root PE
	if (my_pe == root_pe) {
		std::vector<float> host_data(num_floats, 123.456f);
		cudaMemcpy(data_ptr, host_data.data(), size, cudaMemcpyHostToDevice);
		cout << "PE " << root_pe << ": Broadcasting data (value: 123.456)" << endl;
	} else {
		check_cuda_error(cudaMemset(data_ptr, 0, size));
	}
	
	nvshmem_barrier_all();
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	nvshmem_broadcast_kernel<<<1, 1>>>(data_ptr, num_floats, root_pe);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float elapsed_ms;
	cudaEventElapsedTime(&elapsed_ms, start, stop);
	
	// Verify on all PEs
	float first_value;
	cudaMemcpy(&first_value, data_ptr, sizeof(float), cudaMemcpyDeviceToHost);
	cout << "PE " << my_pe << ": Received value: " << first_value 
	     << " (Time: " << elapsed_ms << " ms)" << endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	nvshmem_barrier_all();
}

int main(int argc, char *argv[]) {
	// Initialize NVSHMEM
	nvshmem_init();
	
	int my_pe = nvshmem_my_pe();
	int n_pes = nvshmem_n_pes();
	
	if (n_pes < 2) {
		if (my_pe == 0) {
			cout << "Error: This test requires at least 2 PEs" << endl;
		}
		nvshmem_finalize();
		return 1;
	}
	
	if (my_pe == 0) {
		cout << "=== NVSHMEM Advanced Features Test ===" << endl;
		cout << "Number of PEs: " << n_pes << endl;
		cout << "CUDA Device: " << nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) << endl;
	}
	
	// Allocate symmetric memory
	size_t test_size = 64 * 1024 * 1024;  // 64MB
	float* data_ptr = (float*)nvshmem_malloc(test_size);
	float* counter_ptr = (float*)nvshmem_malloc(sizeof(float));
	uint64_t* signal_ptr = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
	
	if (!data_ptr || !counter_ptr || !signal_ptr) {
		fprintf(stderr, "PE %d: Failed to allocate symmetric memory\n", my_pe);
		nvshmem_finalize();
		return 1;
	}
	
	nvshmem_barrier_all();
	
	// Run tests
	if (my_pe == 0) {
		// Test 1: Direct memory access
		test_direct_access(data_ptr, data_ptr, 1, test_size);
		nvshmem_barrier_all();
		
		// Test 2: Atomic operations
		test_atomic_operations(counter_ptr, 1);
		nvshmem_barrier_all();
		
		// Test 3: Signal/Wait
		test_signal_wait(signal_ptr);
		
		// Test 4: Broadcast
		test_broadcast(data_ptr, test_size);
		
	} else if (my_pe == 1) {
		test_direct_access(data_ptr, data_ptr, 0, test_size);
		nvshmem_barrier_all();
		
		test_atomic_operations(counter_ptr, 0);
		nvshmem_barrier_all();
		
		test_signal_wait(signal_ptr);
		test_broadcast(data_ptr, test_size);
		
	} else {
		// Other PEs just participate in collective operations
		nvshmem_barrier_all();
		nvshmem_barrier_all();
		nvshmem_barrier_all();
		test_broadcast(data_ptr, test_size);
	}
	
	if (my_pe == 0) {
		cout << "\n=== All tests completed ===" << endl;
	}
	
	// Cleanup
	nvshmem_free(data_ptr);
	nvshmem_free(counter_ptr);
	nvshmem_free(signal_ptr);
	nvshmem_finalize();
	
	return 0;
}

