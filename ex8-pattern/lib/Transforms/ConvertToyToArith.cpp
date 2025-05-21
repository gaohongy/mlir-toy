// 在上一个 ex，我们只是简单将 pass 这个流程顺利搭建起来，但是这一 pass 的目标是将 toy IR convert 到 arith IR，也就是 runOnOperation 过程需要真的进行 op 操作了

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/PatternMatch.h"   // mlir::OpRewritePattern
#include "mlir/Dialect/Arith/IR/Arith.h"    // AddOpPat rewrite pattern
#include "mlir/Transforms/DialectConversion.h" // mlir::ConversionTarget
#include <iostream>

#define GEN_PASS_DEF_CONVERTTOYTOARITH
#include "Toy/Passes.h"

// rewrite pattern 定义
// 将 toy 中的多输入加法，lower 到 arith 中的多个 binary add 的链式组合
struct AddOpPat: mlir::OpRewritePattern<toy::AddOp> {
    // 继承基类的构造函数，可通过 patterns.add<AddOpPat>(context); 直接构造 AddOpPat 对象
    using mlir::OpRewritePattern<toy::AddOp>::OpRewritePattern;

    // 因为我们在写 toy::AddOp 时，给定的是let arguments = (ins Variadic<AnyInteger>:$inputs);，即支持变量的累积加法，但是 arith::AddIOp 只支持两变量加法
    mlir::LogicalResult matchAndRewrite(toy::AddOp op, mlir::PatternRewriter &rewriter) const override {
        llvm::SmallVector<mlir::Value> inputs = llvm::to_vector(op.getInputs());
        // ? 为什么是 inputs[0]
        mlir::Value result = inputs[0];

        // ? 这个不是很懂，按理说对于一个 toy::AddOp 只需要创建一个 arith::AddIOp 即可，这里为何多次生成
        for (size_t i = 1; i < inputs.size(); i++) {
            result = rewriter.create<mlir::arith::AddIOp>(op->getLoc(), result, inputs[i]);
        }

        rewriter.replaceOp(op, mlir::ValueRange(result));

        return mlir::success();
    }
};

// 这里用到了 CRTP 实现了静态多态
struct ConvertToyToArithPass : 
    toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>
{
    using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;

    // 此虚函数来自 ConvertToyToArithBase -> OperationPass（final 表示不再允许重写）
    void runOnOperation() final {
        std::cout << "before convert:" << std::endl;

        getOperation()->print(llvm::errs());

        // 合法性声明
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::arith::ArithDialect>();

        // 注册 AddOpPat rewrite pattern
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<AddOpPat>(&getContext());

        if (
            mlir::failed(
                // 执行 Partial Conversion，将所有标记为合法的完成转换，允许保留未标识或者标识为非法的
                mlir::applyPartialConversion(getOperation(), target, std::move(patterns))
            )
        ) {
            signalPassFailure();
        }
        
        // code_name 是 ConvertToyToArithBase 类的一个成员变量，因此可以直接访问
        //llvm::errs() << "get name: " << code_name << "\n";
    }

    // 声明 Pass 所依赖的 dialect，保证在 pass pipeline 开始前完成相关 dialect 的加载
    // 目前理解它完成的就是加载 arith dialect(getContext().loadDialect<mlir::arith::ArithDialect>();)，区别在于这里仅仅是声明依赖关系，实际的加载是 MLIR 框架负责实现的，能够保证是线程安全的
    // 这个函数可以通过在 td 中添加 dependentDialects 由 MLIR 框架自动生成
    //void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    //    registry.insert<mlir::arith::ArithDialect>();
    //}
};

// Passes.h 中的声明的实现
std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass(ConvertToyToArithOptions options) {
    return std::make_unique<ConvertToyToArithPass>(options);
}
