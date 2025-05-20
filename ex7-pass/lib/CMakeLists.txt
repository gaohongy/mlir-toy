# 依赖目标 MLIRToyOpsIncGen（tablegen 生成的相关文件） 生成 libMLIRToy.a
add_mlir_library(MLIRToy toy.cpp DEPENDS MLIRToyOpsIncGen)

add_subdirectory(Transforms)