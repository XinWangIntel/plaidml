#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/Pass/Pass.h"

namespace pmlc::target::nvidia_gpu {

#define GEN_PASS_CLASSES
#include "pmlc/target/nvidia_gpu/passes.h.inc"

} // namespace pmlc::target::nvidia_gpu
