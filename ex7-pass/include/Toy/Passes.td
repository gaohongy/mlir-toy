#ifndef TOY_PASSES_TD
#define TOY_PASSES_TD

include "mlir/Pass/PassBase.td"

// convert-toy-to-arith 将作为命令行工具的 option 选项
def ConvertToyToArith : Pass<"convert-toy-to-arith"> {
    let summary = "Convert Toy to Arith";
    let constructor = "toy::createConvertToyToArithPass()";

    // 为 pass 添加参数，从代码上来说就是为类添加了一个带参构造函数
    let options = [
        // code_name 是 tabelgen 生成的代码中的变量名称
        // cli_name 和 help 是 cli 工具所使用的选项中的名称
        // --convert-toy-to-arith                         -   Convert Toy to Arith
        // --cli_name=<string>                            - help
        Option<"code_name", "cli_name", "std::string", "", "help">
    ];
}

// DCE 的原理是检测 func.return operation，进而根据数据流反向遍历，所以这里可以只约束 pass 仅在 FuncOp 上作用
//def DCE : Pass<"toy-dce", "toy::FuncOp"> {
def DCE : Pass<"toy-dce"> {
    let summary = "dead code elimination";
    let constructor = "toy::createDCEPass()";
}

#endif
