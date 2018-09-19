#include <cuda_runtime.h>
#include <stdio.h>

class CudaChecker {
	public:
	    CudaChecker();
		~CudaChecker();
  
        int initCuda(const int max_device_count_to_use);
};
