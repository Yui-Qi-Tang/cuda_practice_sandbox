#include "cuda_op.h"

// Constructor
CudaChecker::CudaChecker(const int devices) {
	_usedDevices = devices;
	_deviceCounts = deviceCounts();

	if(_usedDevices > _deviceCounts) {
        printf("These are not %d devices(Max devices: %d)", _usedDevices, _deviceCounts);
	}
	getDeviceProps();
    // printf("Initial CUDA ")
	
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


/*Get device counts on your platform*/
int CudaChecker::deviceCounts() {
	int deviceCounts = 0;
	int cudaErrCode = -1; // not set
	cudaErrCode = cudaGetDeviceCount(&deviceCounts);
	if (cudaErrCode != cudaSuccess) {
		fprintf(stderr, "There is no device. and cuda error code is: %d\n", cudaErrCode);
        exit(-1);
	}	
	return deviceCounts;
}

int CudaChecker::getMaxDeviceCounts() {
	return _deviceCounts;
}

void CudaChecker::getDeviceProps() {
	_devProp = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp) * _usedDevices);
	// cuda device 由 0 開始計算，第一個裝置在0的位置
	for(int i = 0; i < _usedDevices; i++) {
        cudaError_t status = cudaGetDeviceProperties((_devProp+i), i);
		if(status != cudaSuccess) {
            printf("Fatal Error: Get device properites failed!\n");
			exit(-1);
		} // fi
	} // for
}

void CudaChecker::dumpDevicesProperty() {
	cudaDeviceProp *devPtr;
	for(int i = 0; i < _usedDevices; i++) {
		devPtr = _devProp + i;
		printf("==>Property of Device no.%d\n",i);
		printf("major: %d\n", devPtr -> major);
		printf("name: %s\n", devPtr -> name);
	}
}
