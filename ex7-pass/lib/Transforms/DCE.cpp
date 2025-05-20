#include "llvm/Support/raw_ostream.h"

#include <iostream>

#define GEN_PASS_DEF_DCE
#include "Toy/Passes.h"

// dead code elimination 实现原理：从返回值出发追溯其依赖的操作（operations），把无用的操作删除掉(标记-清除（mark-sweep）策略)
struct DCEPass : toy::impl::DCEBase<DCEPass> {
    // 应该是递归枚举所有 operation
    void visitAll(llvm::DenseSet<mlir::Operation*> &visited, mlir::Operation *op) {
        if(visited.contains(op)) {
            return;
        }

        visited.insert(op);

        // 保留层次结构中的节点（这些节点可能并不在数据流图中）
        for (mlir::Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
            llvm::outs() << parent->getName() << " -> " << op->getName() << "\n";
            visited.insert(parent);
        }

        // 保留数据流图中的节点
        for (auto operand : op->getOperands()) {
            // 根据 SSA 的 def-use 链，数据流图上反向遍历
            if (mlir::Operation *def = operand.getDefiningOp()) {
                llvm::outs() << def->getName() << " found" << "\n";
                visitAll(visited, def);
            }
        }
    }

    void runOnOperation() final {
        // 从 return 语句反向遍历，标记有效语句
        llvm::DenseSet<mlir::Operation*> visited;
        // 遍历 IR 中的所有 operation，只对 return op 调用 visitAll 函数
        getOperation()->walk([&](toy::ReturnOp op) {
            llvm::outs() << op.getOperation()->getName() << " found" << "\n"; 
            visitAll(visited, op);
        });

        // 再一次遍历，标记无效语句
        llvm::SmallVector<mlir::Operation*> opToRemove;
        getOperation()->walk([&](mlir::Operation *op) {
            llvm::outs() << op->getName() << " " << getOperation()->getName() << "\n";
            if (op == getOperation()) {
                return;
            }
            if (!visited.contains(op)) {
                llvm::outs() << op->getName() << " no use" << "\n";
                opToRemove.push_back(op);
            }
        });

        // 删除无效语句
        for (auto v : reverse(opToRemove)) {
            v->erase();
        }
    }
};

std::unique_ptr<mlir::Pass> toy::createDCEPass() {
    return std::make_unique<DCEPass>();
}
