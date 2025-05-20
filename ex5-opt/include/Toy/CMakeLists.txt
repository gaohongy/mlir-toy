# first Toy: Toy.td; second Toy: dialect name(let name = "toy";)
#ToyDialect.h.inc; ToyDialect.cpp.inc; ToyOps.h.inc; ToyOps.cpp.inc
# 目前的理解是 dialect 和 op 的相关 .h 和 .cpp 是会由这条命令自动生成的，如果还需要其他的内容，需要手动指定
# 从 make help 可以看出，这个指令将生成 MLIRxxxIncGen 这个 .PHONY(xxx 即第一个参数)
add_mlir_dialect(ToyOps toy)

# 从官方的文件来看，这里也是使用的 Ops
# 能够找到相关文件的核心是 mlir-toy 中的 include 设置

# 按照目前的理解，上面一条等价于下面这些内容
#set(LLVM_TARGET_DEFINITIONS Toy.td)
#mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
#mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)
#add_public_tablegen_target(ToyDialectIncGen)
#
#set(LLVM_TARGET_DEFINITIONS Toy.td)
#mlir_tablegen(ToyOps.h.inc -gen-op-decls)
#mlir_tablegen(ToyOps.cpp.inc -gen-op-defs)
#add_public_tablegen_target(ToyOpsIncGen)

# 之所以都使用 Toy.td，是因为它包含了 Dialect 和 op 在内的所有内容。也可以把 ToyDialect.td 和 ToyOps.td 手动写在同一个文件中，不过这样的话结构性较差
# 不过 MLIR 项目模块化极强，一个大型系统可能包含上百个 .td 文件，只有显式用到的 .td 文件才会触发代码生成，从而 提升 CMake 执行效率，避免不必要的构建和生成

# header 将成为 Makefile 下的一个 .PHONY，依赖于 MLIRToyIncGen 这个目标
#add_custom_target(header DEPENDS MLIRToyIncGen)