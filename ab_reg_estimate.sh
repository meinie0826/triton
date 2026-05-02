#!/bin/bash
#=============================================================================
# A/B test for register estimation optimization.
# Applies patch → builds → benchmarks → reverts → builds → benchmarks → compares.
#
# Usage on B300:
#   cd /workspace/triton
#   bash ab_reg_estimate.sh
#
# Prerequisites: Triton cloned + build dependencies installed + main branch clean.
#=============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_DIR="$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/ab_results"
PATCH_FILE="$SCRIPT_DIR/.reg_estimate.patch"
BENCH_SCRIPT="/tmp/benchmark_reg_estimate.py"

mkdir -p "$RESULTS_DIR"

echo "=== Register Estimation A/B Benchmark ==="
echo ""

# ─── Save the benchmark Python script ───
if [ -f "$SCRIPT_DIR/benchmark_reg_estimate.py" ]; then
    cp "$SCRIPT_DIR/benchmark_reg_estimate.py" "$BENCH_SCRIPT"
    echo "Benchmark script saved to $BENCH_SCRIPT"
else
    echo "ERROR: benchmark_reg_estimate.py not found in $SCRIPT_DIR"
    echo "Make sure you are on the reg-estimation branch or have the script."
    exit 1
fi

# ─── Write the patch ───
cat > "$PATCH_FILE" << 'ENDOFPATCH'
diff --git a/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp b/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp
index 78fa16506e..41d98a18b0 100644
--- a/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp
+++ b/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp
@@ -140,11 +140,8 @@ static LogicalResult relayoutWarps(ModuleAxisInfoAnalysis &axisInfo,
 // optimizePartitionWarps
 //===----------------------------------------------------------------------===//

-// Get the number of i32 registers required to store a tensor.
-static unsigned getTensorNumI32Regs(RankedTensorType ty) {
-  unsigned numElems = getTotalElemsPerThread(ty) *
-                      product(getThreadsPerWarp(ty)) *
-                      product(getWarpsPerCTA(ty));
+static unsigned getTensorPerThreadRegs(RankedTensorType ty) {
+  unsigned numElems = getTotalElemsPerThread(ty);
   unsigned elSize =
       isa<PointerType>(ty.getElementType()) ? 64 : ty.getElementTypeBitWidth();
   return numElems * elSize / 32;
@@ -153,25 +150,32 @@ static unsigned getTensorNumI32Regs(RankedTensorType ty) {
 static LogicalResult optimizePartitionNumWarps(ModuleAxisInfoAnalysis &axisInfo,
                                                WarpSpecializeOp wsOp,
                                                RunPipelineFn runPipeline) {
-  // Extremely rough estimate of the number of registers needed per partition.
-  // For each partition, get the number of i32 registers used by the largest
-  // tensor value.
-  //
-  // Because the partition region is isolated from above, we could in theory
-  // compile it to PTX and read the number of registers that got allocated.
-  SmallVector<unsigned> maxTensorRegs;
+  // Estimate the peak per-thread register pressure per partition.
+  // For warp count reduction, we need the per-thread register count of the
+  // largest tensor (when warps are halved, each thread's share doubles).
+  // For requested registers, we sum per-thread regs of all tensors to
+  // account for multiple simultaneously-live values (accumulator + buffers).
+  SmallVector<unsigned> maxPerThreadRegs;
+  SmallVector<unsigned> sumPerThreadRegs;
   for (Region *partition : wsOp.getPartitionRegions()) {
-    unsigned &tensorRegs = maxTensorRegs.emplace_back(0);
+    unsigned &maxRegs = maxPerThreadRegs.emplace_back(0);
+    unsigned &sumRegs = sumPerThreadRegs.emplace_back(0);
+    // Track which tensor types we've already counted to avoid duplicates
+    // (same tensor appearing as both operand and result).
+    DenseSet<RankedTensorType> seen;
     partition->walk([&](Operation *op) {
       for (Type type :
            llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
-        if (auto tensor = dyn_cast<RankedTensorType>(type))
-          tensorRegs = std::max(tensorRegs, getTensorNumI32Regs(tensor));
+        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
+          unsigned regs = getTensorPerThreadRegs(tensor);
+          maxRegs = std::max(maxRegs, regs);
+          if (!seen.contains(tensor)) {
+            seen.insert(tensor);
+            sumRegs += regs;
+          }
+        }
       }
     });
-    // Assume that the largest tensor accounts for half of the registers used
-    // by a warpgroup.
-    tensorRegs *= 2;
   }

   // Reduce the number of warps used by partitions. For partitions with no
@@ -235,13 +239,14 @@ static LogicalResult optimizePartitionNumWarps(ModuleAxisInfoAnalysis &axisInfo,
     int32_t curTotalNumWarps = std::accumulate(
         partitionNumWarps.begin(), partitionNumWarps.end(), defaultNumWarps);

-    for (auto [minWarps, numWarps, tensorRegs] :
-         llvm::zip(minWarpsForPartition, partitionNumWarps, maxTensorRegs)) {
+    for (auto [minWarps, numWarps, maxRegs] :
+         llvm::zip(minWarpsForPartition, partitionNumWarps, maxPerThreadRegs)) {
       if (numWarps <= minWarps)
         continue;
-      // Check if reducing the number of warps will still fit the tensor. If it
-      // didn't fit to begin with, it won't fit after shrinking.
-      unsigned reqRegsPerThread = tensorRegs / threadsPerWarp / (numWarps / 2);
+      // When halving the number of warps, each thread will hold twice as many
+      // elements from the largest tensor. Check that this still fits within
+      // the register budget.
+      unsigned reqRegsPerThread = maxRegs * (numWarps / (numWarps / 2));
       unsigned nextTotalNumWarps = curTotalNumWarps - (numWarps / 2);
       unsigned nextRegsPerThread =
           nTotalRegs / threadsPerWarp / nextTotalNumWarps;
@@ -254,15 +259,20 @@ static LogicalResult optimizePartitionNumWarps(ModuleAxisInfoAnalysis &axisInfo,
   } while (changed);

   SmallVector<int32_t> estRegUsage(partitionNumWarps.size());
-  for (auto [partition, newNumWarps, prevNumWarps, tensorRegs, estRegs] :
+  for (auto [partition, newNumWarps, prevNumWarps, maxRegs, sumRegs, estRegs] :
        llvm::zip(wsOp.getPartitionRegions(), partitionNumWarps,
-                 wsOp.getPartitionNumWarps(), maxTensorRegs, estRegUsage)) {
-    // "Guess" the register usage for each partition.
-    estRegs = tensorRegs ? 88 : 24;
+                 wsOp.getPartitionNumWarps(), maxPerThreadRegs,
+                 sumPerThreadRegs, estRegUsage)) {
+    // Estimate register usage based on the sum of per-thread register counts
+    // of all distinct tensors in the partition. This accounts for multiple
+    // simultaneously-live values (e.g., accumulator + input buffers in
+    // pipelined matmul) without relying on an arbitrary multiplier on the
+    // max tensor.
+    estRegs = sumRegs ? std::max<int32_t>(sumRegs, 40) : 24;

     // Layouts need to be reassigned if the number of warps changed and there
     // are tensor computations.
-    if (newNumWarps == prevNumWarps || !tensorRegs)
+    if (newNumWarps == prevNumWarps || !maxRegs)
       continue;
     // We need to reassign layouts.
     if (failed(relayoutWarps(axisInfo, partition, prevNumWarps, newNumWarps,
ENDOFPATCH
echo "Patch written: $PATCH_FILE"
echo ""

# ─── Helper: build Triton ───
build_triton() {
    echo ">>> Building Triton..."
    cd "$TRITON_DIR"
    BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())' 2>/dev/null || echo "")
    if [ -z "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
        echo "ERROR: build directory not found."
        exit 1
    fi
    ninja -C "$BUILD_DIR" 2>&1 | tail -5
    echo ">>> Build done."
}

# ─── Helper: run benchmark ───
run_bench() {
    local label="$1"
    local output="$RESULTS_DIR/results_${label}.json"
    echo ">>> Clearing Triton cache..."
    rm -rf ~/.triton/cache
    echo ">>> Running benchmark for: $label"
    python3 "$BENCH_SCRIPT" --output "$output" --dump-ir
    echo ">>> Done: $output"
}

# ─── Step 1: Apply patch (AFTER) ───
echo "=========================================================="
echo "  STEP 1: Patched (AFTER)"
echo "=========================================================="
cd "$TRITON_DIR"
git checkout origin/main
git apply "$PATCH_FILE"
build_triton
run_bench "after"

# ─── Step 2: Revert patch (BEFORE) ───
echo ""
echo "=========================================================="
echo "  STEP 2: Clean main (BEFORE)"
echo "=========================================================="
cd "$TRITON_DIR"
git checkout -- .
build_triton
run_bench "before"

# ─── Step 3: Compare ───
echo ""
echo "=========================================================="
echo "  STEP 3: Comparison"
echo "=========================================================="
python3 -c "
import json
b = json.load(open('$RESULTS_DIR/results_before.json'))
a = json.load(open('$RESULTS_DIR/results_after.json'))

print(f'{\"Shape\":<15} {\"Before\":>9} {\"After\":>9} {\"Delta%\":>7} {\"B/cuBL%\":>8} {\"A/cuBL%\":>8} {\"reqRegs_B\":>16} {\"reqRegs_A\":>16}')
print('-'*105)
for bi, ai in zip(b, a):
    n = bi['name']
    bt, at = bi['triton_tflops'], ai['triton_tflops']
    d = (at-bt)/bt*100 if bt else 0
    bc = bi.get('ratio_pct', 0)
    ac = ai.get('ratio_pct', 0)
    br = bi.get('requested_registers','N/A')
    ar = ai.get('requested_registers','N/A')
    print(f'{n:<15} {bt:>9.2f} {at:>9.2f} {d:>6.1f}% {bc:>7.1f}% {ac:>7.1f}% {str(br):>16} {str(ar):>16}')

ds = [a['triton_tflops']-b['triton_tflops'] for b,a in zip(b,a)]
avg = sum(d/bb['triton_tflops'] for d,bb in zip(ds,b))/len(ds)*100
print(f'Average delta: {avg:+.1f}%')
"

echo ""
echo "DONE. Results in $RESULTS_DIR/"
