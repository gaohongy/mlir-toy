#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/raw_ostream.h"

int indent = 0;

llvm::raw_ostream &printIndent() {
  for (int i = 0; i < indent; ++i)
    llvm::outs() << "  ";
  return llvm::outs();
}

int pushIndent() {
    indent += 4;
    return indent;
}

void printOperation(mlir::Operation *op);
void printRegion(mlir::Region &region);
void printBlock(mlir::Block &block);

int main(int argc, char** argv) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect>();

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

    // 三种示例展示新增一个 block 并且添加 operation 的流程，前两种采用了不同的方式将 block 添加到 region 中，最后一种为错误示例
    #ifdef create_block_directly
    mlir::Block *newBlock = new mlir::Block();
    func.getBody().push_back(newBlock);

    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), newBlock);

    // create func.return operation
    builder.setInsertionPointToEnd(newBlock);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({addi}));
    #endif

    #ifdef direct_association_region
    mlir::Block *newBlock = func.addBlock();
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), newBlock);

    // create func.return operation
    builder.setInsertionPointToEnd(newBlock);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({addi}));
    #endif

    #ifdef error
    // MLIR 采用的是显式控制流图，按照下面的写法，第二个 block 不会按顺序被执行到
    mlir::Block *newBlock = func.addBlock();
    builder.setInsertionPointToEnd(newBlock);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({addi}));
    #endif

    // create return operation
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({addi}));
    
    // print mlir
    mod->print(llvm::outs());

    printOperation(mod.getOperation());
    //printOperation(addi.getOperation());

    // 通过这一遍历可以看到 operation 间连接的这种层级关系，不过顺序没太搞懂
    mod.walk([](mlir::Operation *child) {
        llvm::outs() << "child op: " << child->getName() << "\n";
    });

    return 0;
}

void printOperation(mlir::Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";

    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";

      for (mlir::NamedAttribute attr : op->getAttrs()) {
        printIndent() << " - '" << attr.getName() << "' : '"
                      << attr.getValue() << "'\n";
      }
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    if (op->getNumRegions()) {
        auto indent = pushIndent();
        for (mlir::Region &region : op->getRegions())
          printRegion(region);
    }
}

void printRegion(mlir::Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";

    auto indent = pushIndent();
    for (mlir::Block &block : region.getBlocks())
      printBlock(block);
}

void printBlock(mlir::Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // A block main role is to hold a list of Operations: let's recurse into
    // printing each operation.
    auto indent = pushIndent();
    for (mlir::Operation &op : block.getOperations())
      printOperation(&op);
}