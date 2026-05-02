#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// relayoutWarps
//===----------------------------------------------------------------------===//

using RunPipelineFn = function_ref<LogicalResult(OpPassManager &, ModuleOp)>;

// Take the body of a partition into a new `tt.func`. We can use this to run a
// full compiler pipeline on the partition.
static OwningOpRef<ModuleOp> takeIntoFunction(ModuleAxisInfoAnalysis &axisInfo,
                                              Region *partition, int numWarps) {
  // Forward the module attributes (target, number of threads per warp, etc.)
  // onto the container module.
  ModuleOp mod = axisInfo.getModuleOp();
  OwningOpRef<ModuleOp> container = ModuleOp::create(mod.getLoc());
  Block *containerBlock = container->getBody();

  auto b = OpBuilder::atBlockBegin(containerBlock);
  FunctionType funcType = b.getFunctionType(partition->getArgumentTypes(), {});
  auto containerFunc = FuncOp::create(b, mod.getLoc(), "container", funcType);
  containerFunc.getBody().takeBody(*partition);
  container.get()->setAttrs(mod->getAttrs());
  container.get()->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));

  // Replace `ttg.warp_return` with `tt.return` to make the IR valid.
  containerFunc.walk([&](WarpReturnOp op) {
    b.setInsertionPoint(op);
    ReturnOp::create(b, op.getLoc());
    op.erase();
  });

  // This should make valid IR.
  if (failed(mlir::verify(*container)))
    llvm::report_fatal_error("expected partition region to make valid IR");

  // Attach axis info properties.
  auto wsOp = partition->getParentOfType<WarpSpecializeOp>();
  auto *funcInfo =
      axisInfo.getFuncData(wsOp->getParentOfType<FunctionOpInterface>());
  assert(funcInfo && "expected to find function axis info");
  for (auto [i, capture] :
       llvm::enumerate(wsOp.getPartitionOp().getExplicitCaptures())) {
    AxisInfo info = funcInfo->lookup(capture);
    containerFunc.setArgAttr(i, "tt.contiguity",
                             b.getI64IntegerAttr(info.getContiguity(0)));
    containerFunc.setArgAttr(i, "tt.divisibility",
                             b.getI64IntegerAttr(info.getDivisibility(0)));
    containerFunc.setArgAttr(i, "tt.constancy",
                             b.getI64IntegerAttr(info.getConstancy(0)));
  }

  return container;
}

// Take the partition body out of the container module and function.
static void extractPartitionBody(OwningOpRef<ModuleOp> container,
                                 Region *partition) {
  auto containerFunc = cast<FuncOp>(container->lookupSymbol("container"));

  // Rewrite the returns.
  containerFunc.walk([](ReturnOp op) {
    OpBuilder b(op);
    WarpReturnOp::create(b, op.getLoc());
    op.erase();
  });

  partition->takeBody(containerFunc.getBody());
}

// Reset the layouts of operations in a region and re-run layout assignment.
static LogicalResult relayoutWarps(ModuleAxisInfoAnalysis &axisInfo,
                                   Region *partition, int prevNumWarps,
                                   int newNumWarps, RunPipelineFn runPipeline) {
  OwningOpRef<ModuleOp> container =
      takeIntoFunction(axisInfo, partition, prevNumWarps);

  // Start by removing all tensor encodings.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [](RankedTensorType ty) { return ty.cloneWithEncoding({}); });
  // But don't remove them from the tensors inside descriptors.
  replacer.addReplacement([](TensorDescType ty) -> std::pair<Type, WalkResult> {
    return {ty, WalkResult::skip()};
  });
  replacer.recursivelyReplaceElementsIn(*container, /*replaceAttrs=*/false,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  ModuleOp mod = axisInfo.getModuleOp();
  auto target = mod->getAttrOfType<StringAttr>(AttrTargetName);
  if (!target)
    return mlir::emitError(mod.getLoc(), "module missing target specification");
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = TritonGPUDialect::getNumCTAs(mod);

  // Enable `convert-triton-to-tritongpu` to rematerialize source layouts for
  // TTG dialect operations. They will get cleared later.
  OpPassManager pm;
  pm.addPass(
      createConvertTritonToTritonGPU({target.str(), newNumWarps, threadsPerWarp,
                                      numCTAs, /*enableSourceRemat=*/true}));
  pm.addPass(createRelayoutTritonGPU());
  if (failed(runPipeline(pm, *container)))
    return failure();
  // Clear source rematerializations by propagating the source layout.
  container->walk([](UnrealizedConversionCastOp op) {
    op.getResult(0).replaceAllUsesWith(op.getOperand(0));
    op.erase();
  });

  pm.clear();
  pm.addPass(createTritonGPUCoalesce());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonGPUOptimizeThreadLocality());
  pm.addPass(createTritonGPUAccelerateMatmul());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  if (failed(runPipeline(pm, *container)))
    return failure();

  extractPartitionBody(std::move(container), partition);
  return success();
}

//===----------------------------------------------------------------------===//
// optimizePartitionWarps
//===----------------------------------------------------------------------===//

static unsigned getTensorPerThreadRegs(RankedTensorType ty) {
  unsigned numElems = getTotalElemsPerThread(ty);
  unsigned elSize =
      isa<PointerType>(ty.getElementType()) ? 64 : ty.getElementTypeBitWidth();
  return numElems * elSize / 32;
}

static LogicalResult optimizePartitionNumWarps(ModuleAxisInfoAnalysis &axisInfo,
                                               WarpSpecializeOp wsOp,
                                               RunPipelineFn runPipeline) {
  // Estimate the peak per-thread register pressure per partition.
  // For warp count reduction, we need the per-thread register count of the
  // largest tensor (when warps are halved, each thread's share doubles).
  // For requested registers, we sum per-thread regs of all tensors to
  // account for multiple simultaneously-live values (accumulator + buffers).
  SmallVector<unsigned> maxPerThreadRegs;
  SmallVector<unsigned> sumPerThreadRegs;
  SmallVector<bool> hasMMAOp;
  for (Region *partition : wsOp.getPartitionRegions()) {
    unsigned &maxRegs = maxPerThreadRegs.emplace_back(0);
    unsigned &sumRegs = sumPerThreadRegs.emplace_back(0);
    bool &hasDot = hasMMAOp.emplace_back(false);
    // Track which tensor types we've already counted to avoid duplicates
    // (same tensor appearing as both operand and result).
    DenseSet<RankedTensorType> seen;
    partition->walk([&](Operation *op) {
      // Detect DotOpInterface (covers tc_gen5_mma, warp_group_dot, tl.dot).
      // On Blackwell, MMA partitions use MemDescType for operands/results
      // rather than RankedTensorType, so the tensor walk finds nothing.
      // MMA ops need ~40 registers per thread for fragments + pipeline
      // management, which is much higher than the default fallback of 24.
      if (isa<DotOpInterface>(op))
        hasDot = true;
      for (Type type :
           llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
          unsigned regs = getTensorPerThreadRegs(tensor);
          maxRegs = std::max(maxRegs, regs);
          if (!seen.contains(tensor)) {
            seen.insert(tensor);
            sumRegs += regs;
          }
        }
      }
    });
  }

  // Reduce the number of warps used by partitions. For partitions with no
  // tensor computations, always reduce them to 1 warp.
  //
  // We can't use `nvvm.setmaxnreg` because this requires a known value for
  // `maxnreg` on the kernel, which is currently controlled by the frontend.
  // Thus, assume PTXAS will evenly distribute the total pool of registers
  // across all warps.
  //
  // If the compiler could control that, then we could allow non-uniform
  // register distributions, mostly beneficial for single-warp warpgroups that
  // just do some artihmetic.
  constexpr unsigned nTotalRegs = 1 << 16; // for Blackwell SMs
  const unsigned threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(axisInfo.getModuleOp());
  const unsigned defaultNumWarps = lookupNumWarps(wsOp);

  SmallVector<int32_t> partitionNumWarps =
      llvm::to_vector(wsOp.getPartitionNumWarps());

  // Determine if a partition has a lower limit on the number of warps.
  SmallVector<int32_t> minWarpsForPartition(partitionNumWarps.size(), 1);
  for (auto [minWarps, region] :
       llvm::zip(minWarpsForPartition, wsOp.getPartitionRegions())) {
    region->walk([minWarps = &minWarps](Operation *op) {
      // Some instructions have critical throughput if have low register usage.
      // Make sure there are enough warps for these ops to execute quickly.
      // TMAStoreLikeOps stay in the main partition, so they should not appear
      // in partition regions here.
      if (isa<ttng::TMALoadLikeOpInterface>(op))
        *minWarps = 2;
      // TMEM ops require at least 4 warps to be able to read all lanes.
      else if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TMEMAllocOp>(op))
        *minWarps = 4;
    });
  }

  bool changed;
  do {
    changed = false;

    // Assuming even distribution of registers, given the total number of warps
    // currently allocated, we can guess the number of registers PTXAS will
    // distribute to each warp.
    //
    // For example, given 18 warps and a tensor<128x256xf32> contained in an
    // 8-warp partition, we have (nTotalRegs/32/18) = ~113 regs per thread, and
    // the tensor requires 128 regs per thread in its partition. In this case,
    // nothing can be done.
    //
    // However, given a tensor<128x128xf32>, this requires only 64 regs per
    // thread in 8 warps. If we reduce the size of the warp to 4, the overall
    // regs per thread increases to (nTotalRegs/32/14) = ~146 regs per thread,
    // while the tensor now requires 128 regs per thread. This works.
    //
    // The next iteration sees ~170 regs per thread, but the tensor will require
    // 256, which is too many. So the algorithm stops at 4 warps. Evidently, if
    // there are other partitions that can be reduced, we have to iterate this
    // algorithm.
    int32_t curTotalNumWarps = std::accumulate(
        partitionNumWarps.begin(), partitionNumWarps.end(), defaultNumWarps);

    for (auto [minWarps, numWarps, maxRegs] :
         llvm::zip(minWarpsForPartition, partitionNumWarps, maxPerThreadRegs)) {
      if (numWarps <= minWarps)
        continue;
      // When halving the number of warps, each thread will hold twice as many
      // elements from the largest tensor. Check that this still fits within
      // the register budget.
      unsigned reqRegsPerThread = maxRegs * (numWarps / (numWarps / 2));
      unsigned nextTotalNumWarps = curTotalNumWarps - (numWarps / 2);
      unsigned nextRegsPerThread =
          nTotalRegs / threadsPerWarp / nextTotalNumWarps;
      if (reqRegsPerThread <= nextRegsPerThread) {
        numWarps /= 2;
        changed = true;
        break;
      }
    }
  } while (changed);

  SmallVector<int32_t> estRegUsage(partitionNumWarps.size());
  for (auto [partition, newNumWarps, prevNumWarps, maxRegs, sumRegs, hasDot, estRegs] :
       llvm::zip(wsOp.getPartitionRegions(), partitionNumWarps,
                 wsOp.getPartitionNumWarps(), maxPerThreadRegs,
                 sumPerThreadRegs, hasMMAOp, estRegUsage)) {
    // Estimate register usage based on the sum of per-thread register counts
    // of all distinct tensors in the partition. This accounts for multiple
    // simultaneously-live values (e.g., accumulator + input buffers in
    // pipelined matmul) without relying on an arbitrary multiplier on the
    // max tensor.
    // On Blackwell, MMA partitions use MemDescType (TMEM/shared) for
    // operands/results rather than RankedTensorType, so sumRegs=0 and we
    // fall through to the default. MMA ops need ~40 regs/thread for
    // fragments + pipeline management, so bump the estimate for partitions
    // containing DotOpInterface ops.
    if (sumRegs) {
      estRegs = std::max<int32_t>(sumRegs, 40);
    } else if (hasDot) {
      // MMA partition with no RankedTensorType (Blackwell TMEM path).
      // tc_gen5_mma needs ~40 regs/thread for accumulator fragment,
      // A/B fragments, and pipeline management.
      estRegs = 40;
    } else {
      estRegs = 24;
    }

    // Layouts need to be reassigned if the number of warps changed and there
    // are tensor computations or MMA ops.
    if (newNumWarps == prevNumWarps || (!maxRegs && !hasDot))
      continue;
    // We need to reassign layouts.
    if (failed(relayoutWarps(axisInfo, partition, prevNumWarps, newNumWarps,
                             runPipeline)))
      return failure();
  }
  wsOp.setRequestedRegisters(estRegUsage);
  wsOp.setPartitionNumWarps(partitionNumWarps);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUOPTIMIZEPARTITIONWARPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct OptimizePartitionWarps
    : triton::gpu::impl::TritonGPUOptimizePartitionWarpsBase<
          OptimizePartitionWarps> {
  using TritonGPUOptimizePartitionWarpsBase::
      TritonGPUOptimizePartitionWarpsBase;

  void runOnOperation() override;
};
} // namespace

void OptimizePartitionWarps::runOnOperation() {
  SmallVector<WarpSpecializeOp> wsOps;
  getOperation().walk([&](WarpSpecializeOp wsOp) { wsOps.push_back(wsOp); });

  if (wsOps.empty()) {
    return;
  }

  ModuleAxisInfoAnalysis axisInfo(getOperation());
  auto runPipelineFn = [&](OpPassManager &pm, ModuleOp container) {
    // The module must be directly nested under the current op for `runPipeline`
    // to work.
    getOperation().push_back(container);
    llvm::scope_exit remove([&] { container->remove(); });
    return runPipeline(pm, container);
  };

  for (auto wsOp : wsOps) {
    if (failed(optimizePartitionNumWarps(axisInfo, wsOp, runPipelineFn))) {
      return signalPassFailure();
    }
  }
}
