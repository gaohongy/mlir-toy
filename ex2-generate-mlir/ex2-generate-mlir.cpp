#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

// 这部分示例实际会等同于创建 Pass 时对 IR 的修改
int main(int argc, char** argv) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();

    // create builder 
    mlir::OpBuilder builder(&ctx);

    // create model operation
    mlir::ModuleOp mod = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    // set insert point
    builder.setInsertionPointToEnd(mod.getBody());

    // create function operation
    mlir::IntegerType i32 = builder.getI32Type();
    mlir::FunctionType funcType = builder.getFunctionType({i32, i32}, i32);
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test", funcType);

    // create entry block（相较于普通 block 会自动添加参数）
    mlir::Block *entry = func.addEntryBlock();
    mlir::Block::BlockArgListType args = entry->getArguments();

    // set insert point（我理解是，如果不显示指定，并不能确定是插在 model 还是 func operation 中）
    builder.setInsertionPointToEnd(entry);

    // create arith.addi operation
    mlir::arith::AddIOp addi = builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), args[0], args[1]);

    // create func.return operation
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({addi}));
    
    // print mlir
    mod->print(llvm::outs());

    return 0;
}