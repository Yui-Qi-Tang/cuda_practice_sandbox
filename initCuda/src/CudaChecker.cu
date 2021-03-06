#include "CudaChecker.h"

// Constructor
CudaChecker::CudaChecker(const int devices) {
	_usedDevices = devices;
	_deviceCounts = deviceCounts();

	if(_usedDevices > _deviceCounts) {
        printf("These are not %d devices(Max devices: %d)\n", _usedDevices, _deviceCounts);
	}
	getDeviceProps();
	if(!allDevicesSupCuda()) {
		printf("No some deives does not support cuda\n");
		exit(-1);
	}
	printf("Finished check CUDA, There %d device exists\n", _deviceCounts);
	printf("The device number is: ");
	for(int i=0; i < _deviceCounts; i++) {
		printf("%d", i);
	}
	printf("\n");
}

// Destructor
CudaChecker::~CudaChecker() {
	// free pointer since tail to head, avoid pointer broken head
	for(int i=_usedDevices - 1; i >= 0; i--) {
		// printf("relase devPtr[%d]\n", i); // print which pointer is relased
		free((_devProp+i));
	}
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
	/*
		*Use foreach??
		All fields of device property:
	      char name[256];
          size_t totalGlobalMem;
          size_t sharedMemPerBlock;
          size_t memPitch;
          size_t totalConstMem;
		  size_t textureAlignment;
		  

          int regsPerBlock;
          int warpSize;
          int maxThreadsPerBlock;
          int maxThreadsDim[3];
          int maxGridSize[3];
          int major;
          int minor;
          int clockRate;
          int deviceOverlap;
          int multiProcessorCount;
          int kernelExecTimeoutEnabled;
          int integrated;
          int canMapHostMemory;
          int computeMode;
          int concurrentKernels;
          int ECCEnabled;
          int pciBusID;
          int pciDeviceID;
          int tccDriver;
	*/
	for(int i = 0; i < _usedDevices; i++) {
		printf("==>Property of Device no.%d\n", i);
		printf("name: %s\n", (_devProp + i) -> name);
		printf("major: %d\n", (_devProp + i) -> major);
		printf("regsPerBlock: %d\n", (_devProp + i) -> regsPerBlock);
		printf("warpSize: %d\n", (_devProp + i) -> warpSize);
		printf("maxThreadsPerBlock: %d\n", (_devProp + i) -> maxThreadsPerBlock);
		printf("minor: %d\n", (_devProp + i) -> minor);
		printf("clockRate: %d\n", (_devProp + i) -> clockRate);
		printf("deviceOverlap: %d\n", (_devProp + i) -> deviceOverlap);
		printf("multiProcessorCount: %d\n", (_devProp + i) -> multiProcessorCount);
		printf("kernelExecTimeoutEnabled: %d\n", (_devProp + i) -> kernelExecTimeoutEnabled);
		printf("integrated: %d\n", (_devProp + i) -> integrated);
		printf("canMapHostMemory: %d\n", (_devProp + i) -> canMapHostMemory);
		printf("computeMode: %d\n", (_devProp + i) -> computeMode);
		printf("concurrentKernels: %d\n", (_devProp + i) -> concurrentKernels);
		printf("ECCEnabled: %d\n", (_devProp + i) -> ECCEnabled);
		printf("pciBusID: %d\n", (_devProp + i) -> pciBusID);
		printf("pciDeviceID: %d\n", (_devProp + i) -> pciDeviceID);
		printf("tccDriver: %d\n", (_devProp + i) -> tccDriver);
		
		for (int j = 0; j<3; j++) {
		    printf("maxThreadsDim[%d]: %d\n", j, (_devProp + i) -> maxThreadsDim[j] );
		}

		for (int j = 0; j<3; j++) {
		    printf("maxGridSize[%d]: %d\n", j, (_devProp + i) -> maxGridSize[j]);
		}
		
		printf("totalGlobalMem: %lu\n", (_devProp + i) -> totalGlobalMem);
        printf("sharedMemPerBlock: %lu\n", (_devProp + i) -> sharedMemPerBlock);
        printf("memPitch: %lu\n", (_devProp + i) -> memPitch);
        printf("totalConstMem: %lu\n", (_devProp + i) -> totalConstMem);
		printf("textureAlignment: %lu\n", (_devProp + i) -> textureAlignment);
	} // for
}

bool CudaChecker::allDevicesSupCuda() {
	// use major of device property to check
	for(int i = 0; i < _usedDevices; i++) {
	    if( (_devProp + i) -> major < 2 ) {
			return false;
		}
	}
	return true;
}

void CudaChecker::setDevice(int deviceNum) {
	if( deviceNum > (_usedDevices -1) ) {
		printf("Fatal Error: specify an error device number: %d(Max:%d)", deviceNum, (_usedDevices -1));
		exit(-1);
	}

	cudaError_t status = cudaSetDevice(deviceNum);
	if(status != cudaSuccess) {
		printf("Fatal Error: set cuda device execution failed, status: %d\n", status);
		exit(-1);	
	}

	printf("Set cuda device success!!\n");
}