#ifndef __PMLC_TARGET_NVIDIA_GPU_PASSES__
#define __PMLC_TARGET_NVIDIA_GPU_PASSES__

include "mlir/Pass/PassBase.td"

def NvidiaGpuLowerAffine : FunctionPass<"nvidia-gpu-lower-affine"> {
  let summary = "Lower affine -> std + scf, and add GPU mappings ";
  let constructor = "pmlc::target::nvidia_gpu::createNvidiaGpuLowerAffinePass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def AffineIndexPack : FunctionPass<"nvidia-gpu-affine-index-pack"> {
  let summary = "Pack affine parallels with hardware set and > 3 dims into 3 dims";
  let constructor = "pmlc::target::nvidia_gpu::createAffineIndexPackPass()";
}

def ConvertStandardToLLVM : Pass<"nvidia-gpu-convert-std-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Standard + StandardX dialects to LLVM dialect";
  let constructor = "pmlc::target::nvidia_gpu::createConvertStandardToLLVM()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ConvertParallelLoopToGpu : Pass<"nvidia-gpu-convert-parallel-loops-to-gpu"> {
  let summary = "Convert mapped scf.parallel ops to gpu launch operations";
  let constructor = "pmlc::target::nvidia_gpu::createParallelLoopToGpuPass()";
  let dependentDialects = ["mlir::AffineDialect", "mlir::gpu::GPUDialect"];
}

def SubgroupBroadcast : FunctionPass<"stdx-subgroup-broadcast"> {
  let summary = "Perform unvectorization for subgroups broadcast usage";
  let constructor = "pmlc::target::nvidia_gpu::createSubgroupBroadcastPass()";
  let dependentDialects = ["pmlc::dialect::stdx::StdXDialect"];
  let options = [
    Option<"useBlockOps", "use-block-ops", "bool",
           /*default=*/"false",
           "Use intel subgroup block reads and writes">
  ];
}

def ConvertPXAToAffine : Pass<"nvidia-gpu-convert-pxa-to-affine", "mlir::ModuleOp"> {
  let summary = "Convert PXA dialect to Affine dialect";
  let constructor = "pmlc::target::nvidia_gpu::createLowerPXAToAffinePass()";
  let dependentDialects = ["mlir::AffineDialect"];
}

#endif
