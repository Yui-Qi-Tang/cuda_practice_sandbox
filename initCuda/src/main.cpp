#include "cuda_op.h"

int main(int argc, char* argv[]) {
    CudaChecker checker(1);
    printf("Numbers of device: %d", checker.getMaxDeviceCounts());
    //checker.initCuda(1);
    return 0;    
}
