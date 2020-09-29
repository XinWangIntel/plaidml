#pragma once

#include <memory>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::target::nvidia_gpu {

std::unique_ptr<mlir::Pass> createNvidiaGpuLowerAffinePass();

std::unique_ptr<mlir::Pass> createAffineIndexPackPass();

std::unique_ptr<mlir::Pass> createConvertStandardToLLVM();

std::unique_ptr<mlir::Pass> createParallelLoopToGpuPass();

std::unique_ptr<mlir::Pass> createSubgroupBroadcastPass();
std::unique_ptr<mlir::Pass> createSubgroupBroadcastPass(bool useBlockOps);

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

void pipelineBuilder(mlir::OpPassManager &pm);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/target/nvidia_gpu/passes.h.inc"

} // namespace pmlc::target::nvidia_gpu
