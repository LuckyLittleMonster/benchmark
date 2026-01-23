#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::stoi;

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

// Kernel that performs multiple iterations internally to amortize launch overhead
__global__ void memcpy_kernel_float4_multi(float* __restrict__ dst, 
                                            const float* __restrict__ src, 
                                            size_t num_floats,
                                            int num_iterations) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	size_t num_float4 = num_floats / 4;
	
	float4* dst4 = reinterpret_cast<float4*>(dst);
	const float4* src4 = reinterpret_cast<const float4*>(src);
	
	// Perform multiple iterations within single kernel launch
	for (int iter = 0; iter < num_iterations; iter++) {
		// Vectorized copy using float4
		for (size_t i = idx; i < num_float4; i += stride) {
			dst4[i] = src4[i];
		}
		
		// Handle remaining elements
		size_t remaining_start = num_float4 * 4;
		for (size_t i = remaining_start + idx; i < num_floats; i += stride) {
			dst[i] = src[i];
		}
	}
}

// Simple kernel for small sizes
__global__ void memcpy_kernel_simple(float* __restrict__ dst, 
                                      const float* __restrict__ src, 
                                      size_t num_floats) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	
	for (size_t i = idx; i < num_floats; i += stride) {
		dst[i] = src[i];
	}
}

// Original kernel without internal timing
__global__ void memcpy_kernel_float4(float* __restrict__ dst, 
                                      const float* __restrict__ src, 
                                      size_t num_floats) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	size_t num_float4 = num_floats / 4;
	
	float4* dst4 = reinterpret_cast<float4*>(dst);
	const float4* src4 = reinterpret_cast<const float4*>(src);
	
	for (size_t i = idx; i < num_float4; i += stride) {
		dst4[i] = src4[i];
	}
	
	size_t remaining_start = num_float4 * 4;
	for (size_t i = remaining_start + idx; i < num_floats; i += stride) {
		dst[i] = src[i];
	}
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <src_dev> <dst_dev>" << endl;
		return 1;
	}
	int src_dev = stoi(argv[1]);
    int dst_dev = stoi(argv[2]);
	size_t min_size = 1;
	size_t max_size = 4ul * 1024 * 1024 * 1024;
	int iterations = 32;
	int skip = 3;
    
    check_cuda_error(cudaSetDevice(src_dev));
    
    // Get GPU clock rate for converting cycles to time
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, src_dev));
    double clockRateKHz = prop.clockRate;
    cout << "GPU " << src_dev << ": " << prop.name << endl;
    cout << "Clock rate: " << clockRateKHz / 1000.0 << " MHz" << endl;
    
    float *src_ptr;
    check_cuda_error(cudaMalloc(&src_ptr, max_size));

    check_cuda_error(cudaSetDevice(dst_dev));
    float *dst_ptr;
    check_cuda_error(cudaMalloc(&dst_ptr, max_size));
    
    // No need for special timer memory allocations

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

	if (src_dev == dst_dev) {
		cout << "\n=== Single GPU Memory Bandwidth Test (GPU " << src_dev << ") ===" << endl;
	} else {
		check_cuda_error(cudaDeviceEnablePeerAccess(dst_dev, 0));
		cout << "\n=== P2P Bandwidth Test (GPU " << src_dev << " -> GPU " << dst_dev << ") ===" << endl;
	}
	
	cout << "\n--- Method 1: cudaEvent (includes kernel launch overhead) ---" << endl;
	cout << ljust("size", 20) << "\t" << ljust("time(ms)", 20) << "\t" << ljust("bw(GB/s)", 20) << endl;
	
	for (size_t size = min_size; size <= max_size; size *= 2) {
		float total_time = 0.0;
		size_t num_floats = size / sizeof(float);
		
		int blockSize = 256;
		int numBlocks = (num_floats + blockSize - 1) / blockSize;
		numBlocks = (numBlocks > 268435456) ? 268435456 : numBlocks;
		
		// Warm-up
		for (int i = 0; i < skip; i++) {
			if (size >= 16) {
				memcpy_kernel_float4<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats);
			} else {
				memcpy_kernel_simple<<<1, 32>>>(dst_ptr, src_ptr, num_floats);
			}
		}
		
		// Timed with cudaEvent
		check_cuda_error(cudaEventRecord(start_event, 0));
		for (int i = 0; i < (iterations - skip); i++) {
			if (size >= 16) {
				memcpy_kernel_float4<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats);
			} else {
				memcpy_kernel_simple<<<1, 32>>>(dst_ptr, src_ptr, num_floats);
			}
		}
		check_cuda_error(cudaEventRecord(stop_event, 0));
		check_cuda_error(cudaEventSynchronize(stop_event));
		check_cuda_error(cudaEventElapsedTime(&total_time, start_event, stop_event));
		
		double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
		cout << ljust(std::to_string(size), 20) << ljust(std::to_string(total_time), 20) << ljust(std::to_string(bandwidth), 20) << endl;
	}
	
	// Test with amortized launch overhead (single kernel, multiple internal iterations)
	cout << "\n--- Method 2: Amortized Launch Overhead (single kernel call) ---" << endl;
	cout << ljust("size", 20) << "\t" << ljust("time(ms)", 20) << "\t" << ljust("bw(GB/s)", 20) << "\t" << ljust("launch_ovhd(us)", 20) << endl;
	
	for (size_t size = 16; size <= max_size; size *= 2) {
		size_t num_floats = size / sizeof(float);
		
		int blockSize = 256;
		int numBlocks = (num_floats + blockSize - 1) / blockSize;
		numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
		
		// Warm-up
		memcpy_kernel_float4_multi<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, skip);
		check_cuda_error(cudaDeviceSynchronize());
		
		// Single kernel launch performing all iterations internally
		float single_launch_time = 0.0;
		check_cuda_error(cudaEventRecord(start_event, 0));
		memcpy_kernel_float4_multi<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats, iterations - skip);
		check_cuda_error(cudaEventRecord(stop_event, 0));
		check_cuda_error(cudaEventSynchronize(stop_event));
		check_cuda_error(cudaEventElapsedTime(&single_launch_time, start_event, stop_event));
		
		// Multiple kernel launches (from Method 1)
		float multi_launch_time = 0.0;
		check_cuda_error(cudaEventRecord(start_event, 0));
		for (int i = 0; i < (iterations - skip); i++) {
			memcpy_kernel_float4<<<numBlocks, blockSize>>>(dst_ptr, src_ptr, num_floats);
		}
		check_cuda_error(cudaEventRecord(stop_event, 0));
		check_cuda_error(cudaEventSynchronize(stop_event));
		check_cuda_error(cudaEventElapsedTime(&multi_launch_time, start_event, stop_event));
		
		// Calculate bandwidth and launch overhead
		double bandwidth = (size * (iterations - skip) / (1024.0 * 1024.0 * 1024.0)) / (single_launch_time / 1000.0);
		
		// Total launch overhead = difference between multi and single launch
		double total_launch_overhead = multi_launch_time - single_launch_time;
		// Average per launch (in microseconds)
		double avg_launch_overhead_us = (total_launch_overhead / (iterations - skip)) * 1000.0;
		
		cout << ljust(std::to_string(size), 20) 
		     << ljust(std::to_string(single_launch_time), 20) 
		     << ljust(std::to_string(bandwidth), 20)
		     << ljust(std::to_string(avg_launch_overhead_us), 20) << endl;
	}
	
	// Cleanup
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	cudaFree(src_ptr);
	cudaFree(dst_ptr);
	
	return 0;
}

