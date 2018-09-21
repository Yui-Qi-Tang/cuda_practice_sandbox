#include "CudaChecker.h"

int main(int argc, char* argv[]) {
    CudaChecker checker(1);
    checker.dumpDevicesProperty();
    checker.setDevice(0);
    return 0;    
}
