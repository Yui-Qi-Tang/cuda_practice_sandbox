#include <cuda_runtime.h>
#include <stdio.h>

class CudaChecker {
	public:
	    CudaChecker(int devices);
		~CudaChecker();
  
        int initCuda(const int max_device_count_to_use);
		int getMaxDeviceCounts();

	private:
	    int deviceCounts();

	private:
	    int _usedDevices;
		int _deviceCounts;
};
