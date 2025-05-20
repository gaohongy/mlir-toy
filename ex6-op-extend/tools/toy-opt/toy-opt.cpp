#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Dialect 
#include "mlir/IR/DialectRegistry.h"
// Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 自定义 Dialect
#include "Toy/ToyDialect.h"

// MLIR Pass
#include "mlir/Transforms/Passes.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;

    // 注册 dialect
    registry.insert<toy::ToyDialect, mlir::func::FuncDialect>();

    // 注册 pass
    mlir::registerCSEPass();            // Common Subexpression Elimination​​（公共子表达式消除）
    mlir::registerCanonicalizerPass();  // ​​Dead Code Elimination​​（死代码消除）

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "toy-opt", registry));
}