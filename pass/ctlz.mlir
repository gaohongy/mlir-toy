func.func @main(%a : i32) -> i32 {
    %b = math.ctlz %a : i32
    func.return %b : i32
}

// mlir-opt --convert-math-to-funcs=convert-ctlz ctlz.mlir