#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "Toy/ToyDialect.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES

#include "Toy/ToyOps.h.inc"