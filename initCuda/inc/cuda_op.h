#include <cuda_runtime.h>
#include <stdio.h>

class CudaChecker {
	public:
	    CudaChecker(int devices);
		~CudaChecker();

		int getMaxDeviceCounts();
		void setDevice(int deviceNum);

		void dumpDevicesProperty();

	private:
	    int deviceCounts();
		void getDeviceProps();
		bool allDevicesSupCuda();

	private:
	    int _usedDevices;
		int _deviceCounts;
		cudaDeviceProp *_devProp;
};
