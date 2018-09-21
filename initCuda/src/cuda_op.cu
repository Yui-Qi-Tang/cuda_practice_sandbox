#include "cuda_op.h"

// Constructor
CudaChecker::CudaChecker() {
	// This is blank;
}

// Destructor
CudaChecker::~CudaChecker() {
	// This is blank;
}

// initilize CUDA and get device count
int CudaChecker::initCuda(const int max_device_count_to_use){
	int deviceCount=0;
	int cuda_err_code = -1; // not set
	cuda_err_code = cudaGetDeviceCount(&deviceCount);
	if(deviceCount==0){
		fprintf(stderr, "There is no device. and cuda_code is: %d\n", cuda_err_code);
		return 0;
	}
	else {
		fprintf(stdout, "device count:%d\n", deviceCount);
	}
	
	if(deviceCount>max_device_count_to_use){
		fprintf(
			stdout,
			"deviceCount (%d) > max_device_count_to_use (%d), use first some GPUs only.\n",
			deviceCount, max_device_count_to_use
		);
		deviceCount = max_device_count_to_use;
	}
	
	// Check if EVERY devices support CUDA
	bool all_device_sup = true;
	cudaDeviceProp prop;
	for(int i = 0; i < deviceCount; i++) {
		cudaError_t val = cudaGetDeviceProperties(&prop, i);
		if(val != cudaSuccess || prop.major<2){
			all_device_sup = false;
			break;
		}
	}
	
	if(all_device_sup==true){
		fprintf(stdout, "CUDA prop.major:%d\n", prop.major);
	}
	else {
		fprintf(stderr, "Not all devices supporting CUDA.\n");
		return 0;
	}
	
	cudaSetDevice(0);
    printf("cudaSetDevice %d/%d\n", 0, deviceCount);
	return deviceCount;
}
