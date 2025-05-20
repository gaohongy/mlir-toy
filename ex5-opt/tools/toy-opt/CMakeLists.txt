# 类似于 add_executable()，生成可执行文件
add_mlir_tool(toy-opt toy-opt.cpp)

target_link_libraries(toy-opt
    PRIVATE
    MLIRIR MLIRParser MLIRSupport
    MLIRToy    
    MLIROptLib
    MLIRFuncDialect
    MLIRTransforms
)
