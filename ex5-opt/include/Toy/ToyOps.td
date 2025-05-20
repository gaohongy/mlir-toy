#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "Toy/ToyBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// base class
class ToyOp<string mnemonic, list<Trait> traits = []> : 
    Op<ToyDialect, mnemonic, traits>;

// pure trait 会为 op 自动注册 cse 和 dce 模式
def AddOp : ToyOp<"add", [Pure]> {
    let summary = "add operation";
    let arguments = (ins Variadic<AnyInteger>:$inputs);
    let results = (outs AnyInteger:$result);
}

def SubOp : ToyOp<"sub", [Pure]> {
    let summary = "sub operation";
    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger:$result);
}

def ConstantOp : ToyOp<"const", [Pure]> {
    let summary = "const operation";
    let arguments = (ins APIntAttr:$value);
    let results = (outs AnyInteger:$result);
}

#endif
