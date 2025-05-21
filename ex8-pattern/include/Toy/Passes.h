#pragma once

#include "mlir/Pass/Pass.h"
#include "Toy/ToyOps.h"     // pass 在实现时需要访问 operation

// 涉及的宏：
// GEN_PASS_DECL
// GEN_PASS_DECL_CONVERTTOYTOARITH
// GEN_PASS_DEF_CONVERTTOYTOARITH
// GEN_PASS_REGISTRATION
// GEN_PASS_CLASSES(Deprecated)

namespace toy {
    // 虽然 Passes.h.inc 中确实包含这个宏，但是目前的代码 #define 后的内部并没有进行什么有效的处理
    // 加上 options 后声明部分就会出现部分代码了
    #define GEN_PASS_DECL
    // 此宏会 define GEN_PASS_DECL_CONVERTTOYTOARITH
    #include "Toy/Passes.h.inc"

    // 底层实现的接口
    // 在 GEN_PASS_REGISTRATION 宏控制的部分，存在另外的函数会调用这一个函数
    std::unique_ptr<mlir::Pass> createConvertToyToArithPass(ConvertToyToArithOptions options={}); 
    std::unique_ptr<mlir::Pass> createDCEPass();

    // 为上层 opt 工具提供的注册接口
    #define GEN_PASS_REGISTRATION
    #include "Toy/Passes.h.inc"
}
