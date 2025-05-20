cmake_minimum_required(VERSION 3.13.4)

project(mlir-toy VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 便于代码高亮
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "APPEND CMAKE_MODULE_PATH in: ${MLIR_CMAKE_DIR}")
message(STATUS "APPEND CMAKE_MODULE_PATH in: ${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
message(STATUS "Using LLVM_INCLUDE_DIRS in: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")

## ex1 compile
#add_executable(ex1-io ex1-io/ex1-io.cpp)
#target_link_libraries(
#  ex1-io
#  MLIRIR
#  MLIRParser
#  MLIRFuncDialect
#  MLIRArithDialect
#)
#
## ex2 compile
#add_executable(
#    ex2-generate-mlir 
#    ex2-generate-mlir/ex2-generate-mlir.cpp
#    )
#target_link_libraries(
#  ex2-generate-mlir 
#  MLIRIR
#  MLIRParser
#  MLIRFuncDialect
#  MLIRArithDialect
#)

## operation 操纵测试编译
#add_executable(
#    manipulate-operation
#    manipulate-operation/manipulate-operation.cpp
#    )
#target_link_libraries(
#  manipulate-operation
#  MLIRIR
#  MLIRParser
#  MLIRFuncDialect
#  MLIRArithDialect
#  MLIRControlFlowDialect
#)

# ex3 compile
#add_subdirectory(ex3-dialect)

# ex4 为 dialect 和 operation 构建库，可供其他人使用（从 llvm lib 来看，每个 dialect 都有对应的 lib），不过如果只是自己用也可以不创建静态库。但是使用时可以维持统一的链接流程
#add_subdirectory(ex4-lib)

# ex5 opt 工具构建
#add_subdirectory(ex5-opt)

# ex6 完善 operation，添加验证器，指定输出格式
#add_subdirectory(ex6-op-extend)

# ex7 创建 pass
add_subdirectory(ex7-pass)

#add_executable(mlir-toy main.cpp)