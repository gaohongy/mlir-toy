# 类似于 add_executable()，生成可执行文件
add_mlir_tool(toy-opt toy-opt.cpp)

# MLIRToy
# ToyTransforms
target_link_libraries(toy-opt
    PRIVATE
    MLIRIR MLIRParser MLIRSupport
    MLIRToy
    ToyTransforms
    MLIROptLib
    MLIRFuncDialect
    MLIRTransforms
)
