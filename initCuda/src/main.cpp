#include "cuda_op.h"

int main(int argc, char* argv[]) {
    CudaChecker checker(1);
    // printf("Max numbers of device: %d\n", checker.getMaxDeviceCounts());
    //checker.initCuda(1);
    checker.dumpDevicesProperty();

    return 0;    
}
