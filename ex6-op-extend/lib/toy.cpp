#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

// tablegen 生成的 *.cpp.inc
#include "Toy/ToyOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"

void toy::ToyDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "Toy/ToyOps.cpp.inc"
    >();
}

mlir::LogicalResult toy::SubOp::verify() {
    auto lhsType = getLhs().getType();
    auto rhsType = getRhs().getType();

    if (lhsType != rhsType) {
        // emitError 是 AddOp 的成员函数，不同 op 具备自己的成员函数，用于区分哪个 op 发生了错误
        return this->emitError() << "Lhs Type " << lhsType
            << " not equal to rhs " << rhsType;
    }

    return mlir::success();
}

mlir::LogicalResult toy::ConstantOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    Adaptor adaptor,
    llvm::SmallVectorImpl<mlir::Type> &inferedReturnType
) {
    auto type = adaptor.getValueAttr().getType();
    inferedReturnType.push_back(type);
    return mlir::success();
}