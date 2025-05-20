#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "Toy/ToyDialect.h"

// ToyOps 的 td 在定义时就依赖其他的 td，生成的对应的文件也自然依赖其他的头文件
#include "mlir/Interfaces/SideEffectInterfaces.h"   // Pure trait
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // Terminator, ReturnLike trait
#include "mlir/Interfaces/InferTypeOpInterface.h"   // InferTypeOpAdaptor

#define GET_OP_CLASSES

#include "Toy/ToyOps.h.inc"