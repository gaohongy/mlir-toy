# MLIRToyTransformsIncGen 表示的是 pass 相关的 td 文件生成为头文件

add_mlir_library(
    ToyTransforms
    ConvertToyToArith.cpp
    DCE.cpp
    DEPENDS MLIRToyTransformsIncGen
)