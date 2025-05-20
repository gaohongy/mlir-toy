#include "llvm/Support/raw_ostream.h"
#include <iostream>

#define GEN_PASS_DEF_CONVERTTOYTOARITH
#include "Toy/Passes.h"

// 这里用到了 CRTP 实现了静态多态
struct ConvertToyToArithPass : 
    toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>
{
    using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;

    // 此虚函数来自 ConvertToyToArithBase -> OperationPass（final 表示不再允许重写）
    void runOnOperation() final {
        std::cout << "runOnOperation is called" << std::endl;

        getOperation()->print(llvm::errs());

        // code_name 是 ConvertToyToArithBase 类的一个成员变量，因此可以直接访问
        llvm::errs() << "get name: " << code_name << "\n";
    }
};

// Passes.h 中的声明的实现
std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass(ConvertToyToArithOptions options) {
    return std::make_unique<ConvertToyToArithPass>(options);
}
