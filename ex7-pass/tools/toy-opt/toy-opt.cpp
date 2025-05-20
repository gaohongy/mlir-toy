#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Dialect 
#include "mlir/IR/DialectRegistry.h"
// Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 自定义 Dialect
#include "Toy/ToyDialect.h"

// Pass
// MLIR Pass
#include "mlir/Transforms/Passes.h"
// customize toy Pass
#include "Toy/Passes.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;

    // 注册 dialect
    registry.insert<toy::ToyDialect, mlir::func::FuncDialect>();

    // 注册 pass
    mlir::registerCSEPass();            // Common Subexpression Elimination​​（公共子表达式消除）
    mlir::registerCanonicalizerPass();  // ​​Dead Code Elimination​​（死代码消除）
    toy::registerPasses();              // tablegen 生成的固定调用链：registerPasses -> registerConvertToyToArith -> createConvertToyToArithPass

    // MlirOptMain: mlir-opt 工具模板，负责加载 IR，注册 dialect，执行 pass pipeline
    // 简单看了一下 MlirOptMain 的实现(llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp)，还是没有确定是在哪里输出的 MLIR
    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "toy-opt", registry
    ));
}