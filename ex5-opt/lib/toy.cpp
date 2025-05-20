#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

// tablegen 生成的 *.cpp.inc
#include "Toy/ToyOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"


using namespace toy;

void ToyDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "Toy/ToyOps.cpp.inc"
    >();
}
