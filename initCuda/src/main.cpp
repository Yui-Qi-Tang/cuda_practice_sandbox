#include "cuda_op.h"

int main(int argc, char* argv[]) {
    CudaChecker checker;
    checker.initCuda(1);
    return 0;    
}
