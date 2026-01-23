#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::stoi;

string rjust(string s, int width) {
	return string(width - s.size(), ' ') + s;
}
string ljust(string s, int width) {
	return s + string(width - s.size(), ' ');
}

// CUDA kernel for data movement using vectorized load/store
__global__ void memcpy_kernel_float4(float* __restrict__ dst, const float* __restrict__ src, size_t num_floats) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	size_t num_float4 = num_floats / 4;
	
	float4* dst4 = reinterpret_cast<float4*>(dst);
	const float4* src4 = reinterpret_cast<const float4*>(src);
	
	// Vectorized copy using float4
	for (size_t i = idx; i < num_float4; i += stride) {
		dst4[i] = src4[i];
	}
	
	// Handle remaining elements (if num_floats is not divisible by 4)
	size_t remaining_start = num_float4 * 4;
	for (size_t i = remaining_start + idx; i < num_floats; i += stride) {
		dst[i] = src[i];
	}
}

// Simple kernel for small sizes
__global__ void memcpy_kernel_simple(float* __restrict__ dst, const float* __restrict__ src, size_t num_floats) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	
	for (size_t i = idx; i < num_floats; i += stride) {
		dst[i] = src[i];
	}
}

#define check_cuda_error(func) do { \
	func; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


int main(int argc, char *argv[]) {
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <src_dev> <dst_dev>" << endl;
		return 1;
	}
	int src_dev = stoi(argv[1]);
    int dst_dev = stoi(argv[2]);  // Changed to 0: same GPU for memory bandwidth test
	size_t min_size = 1;
	size_t max_size = 4ul * 1024 * 1024 * 1024;
	int iterations = 32;
	int skip = 3;
    
    check_cuda_error(cudaSetDevice(src_dev));
    float *src_ptr;
    check_cuda_error(cudaMalloc(&src_ptr, max_size));

    check_cuda_error(cudaSetDevice(dst_dev));
    float *dst_ptr;
    check_cuda_error(cudaMalloc(&dst_ptr, max_size));

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

	if (src_dev == dst_dev) {
		cout << "=== Single GPU Memory Bandwidth Test (GPU " << src_dev << ") ===" << endl;
	} else {
		check_cuda_error(cudaDeviceEnablePeerAccess(0, 0));
		cout << "=== P2P Bandwidth Test (GPU " << src_dev << " -> GPU " << dst_dev << ") ===" << endl;
	}
	cout << ljust("size", 30) << "\t" << ljust("time(us)", 30) << "\t" << ljust("bw(MB/s)", 30) << endl;
	for (size_t size = min_size; size <= max_size; size *= 2) {
		float total_time = 0.0;
		size_t num_floats = size / sizeof(float);
		
		// Calculate grid and block dimensions
		int blockSize = 256;
		int numBlocks = (num_floats + blockSize - 1) / blockSize;
		// Limit number of blocks for very large transfers
		numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
		
		// Warm-up iterations
		for (int i = 0; i < skip; i++) {
			if (size >= 16) {  // Use vectorized kernel for sizes >= 16 bytes
				memcpy_kernel_float4<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats);
			} else {
				memcpy_kernel_simple<<<1, 32>>>(dst_ptr, src_ptr, num_floats);
			}
		}
		// cudaDeviceSynchronize();
		
		// Record start time before the loop
		cudaEventRecord(start_event, 0);
		for (int i = 0; i < (iterations - skip); i++) {
			if (size >= 16) {  // Use vectorized kernel for sizes >= 16 bytes
				memcpy_kernel_float4<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats);
			} else {
				memcpy_kernel_simple<<<1, 32>>>(dst_ptr, src_ptr, num_floats);
			}
		}
		// Record stop time after the loop
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&total_time, start_event, stop_event);
		
		double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0)) / (total_time / 1000.0);
		cout << ljust(std::to_string(size), 30) << ljust(std::to_string(total_time * 1000.0 / (iterations - skip)), 30)  << ljust(std::to_string(bandwidth), 30) << endl;
	}
	
	// Cleanup
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	cudaFree(src_ptr);
	cudaFree(dst_ptr);
	
	return 0;
}

