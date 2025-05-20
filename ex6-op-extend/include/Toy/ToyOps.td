#ifndef TOY_OPS_TD
#define TOY_OPS_TD

// 官方
include "mlir/IR/OpBase.td"

// 自定义
include "Toy/ToyBase.td"

// 下面使用一些特性时的依赖
include "mlir/Interfaces/SideEffectInterfaces.td"   // Pure trait
include "mlir/Interfaces/ControlFlowInterfaces.td"  // Terminator, ReturnLike trait
include "mlir/Interfaces/InferTypeOpInterface.td"   // InferTypeOpAdaptor

// base class
class ToyOp<string mnemonic, list<Trait> traits = []> : 
    Op<ToyDialect, mnemonic, traits>;

// pure trait 会为 op 自动注册 cse 和 dce 模式
def AddOp : ToyOp<"add", [Pure]> {
    let summary = "add operation";
    let arguments = (ins Variadic<AnyInteger>:$inputs);                             // Variadic: 可变参数
    let results = (outs AnyInteger:$result);
    // %0 = "toy.add"(%arg0, %arg1) : (i32, i32) -> i32
    // %0 = toy.add %arg0, %arg1 : i32, i32 -> i32
    let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";  // 自定义 IR 输出格式
}

def SubOp : ToyOp<"sub", [Pure]> {
    let summary = "sub operation";
    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger:$result);
    let hasVerifier = true;                                                         // 为 AddOp 类添加一个自定义语义约束检查，需要由我们自行实现验证逻辑
}

// 添加 InferTypeOpAdaptor trait
def ConstantOp : ToyOp<"const", [Pure, InferTypeOpAdaptor]> {
//def ConstantOp : ToyOp<"const", [Pure]> {
    let summary = "const operation";
    let arguments = (ins APIntAttr:$value);
    let results = (outs AnyInteger:$result);

    // 添加自定义函数，获取此 op 类型的位宽
    // 按理说这个 getBitWidth 函数的实现是可以放在 cpp 中实现的，不过目前不确定该怎么写
    let extraClassDeclaration = [{
        int64_t getBitWidth() {
            return getResult().getType().getWidth();
        }
    }];
}

def ReturnOp : ToyOp<"return", [Terminator, ReturnLike]> {
    let summary = "return operation";
    let arguments = (ins Optional<AnyInteger>:$data);
}

//def FuncOp: ToyOp<"func", [FunctionOpInterface, RegionKindInterface]> {
//    let summary = "function operation";
//    let arguments = (ins
//        SymbolNameAttr:$sym_name,
//        TypeAttrOf<FunctionType>:$function_type,
//        // FunctionOpInterface 的需要
//        OptionalAttr<DicArrayAttr>:$arg_attrs,
//        OptionalAttr<DicArrayAttr>:$res_attrs
//    );
//}

#endif
