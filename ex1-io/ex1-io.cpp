#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

//using namespace mlir;

int main(int argc, char** argv) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();

    // load file
    mlir::OwningOpRef<mlir::ModuleOp> src = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);

    // output dialect
    src->print(llvm::outs());
    src->dump();

    return 0;
}