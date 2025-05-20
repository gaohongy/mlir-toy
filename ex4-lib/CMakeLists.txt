# 尚不明确作用
#set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})
#message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

# 指定 td 的 include 搜索路径，使得 tablegen 工具可以找到 td 文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "include ${CMAKE_CURRENT_SOURCE_DIR}/include: ${CMAKE_CURRENT_SOURCE_DIR}/include")

# 需要 include tablegen 生成的相关文件
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "include CMAKE_CURRENT_BINARY_DIR/include: ${CMAKE_CURRENT_BINARY_DIR}/include")

add_subdirectory(include/Toy)
add_subdirectory(lib)