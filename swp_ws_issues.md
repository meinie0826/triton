# Triton SWP & Warp Specialization: Current Issues & Improvement Opportunities

## Overview

We investigated Triton's software pipelining (SWP / modulo scheduling) and warp specialization (WS) on Blackwell (B200/SM100). Below is a summary of the key problems we found and the most promising directions for improvement.

---

## Problem 1: `requestedRegisters` has no effect (setmaxnreg not emitted)

**Status: Fix in progress on `reg-estimation` branch**

**Root cause:** `AllocateWarpGroups.cpp` requires `totalPartitionWarps % 4 == 0` to emit `setmaxnreg` PTX instructions. Current WS kernels produce `partitionNumWarps = [1, 2]` (total = 3), so the pass bails out and `requestedRegisters` becomes a dead attribute.

**Fix:** In `OptimizePartitionWarps.cpp`, force MMA partitions (containing `DotOpInterface` ops) to have `minWarps >= 4`. This makes MMA partition a full warp group (128 threads), enabling `setmaxnreg`.

**Expected behavior change:**
- Before: `requestedRegisters = 24, 24`, `setmaxnreg` never emitted
- After: `requestedRegisters = 40, 24`, `setmaxnreg` emitted, MMA partition gets 40 regs, default partition gets leftover

**Risk:** Increasing MMA partition from 1 warp to 4 warps changes occupancy and may affect small-shape performance. Need A/B testing.

---

## Problem 2: SWP distance > 1 not supported (two places)

**Status: Known limitation, not yet addressed**

Both `AssignLatencies.cpp` (preCondition) and `PipelineExpander.cpp` (initializeLoopInfo) bail out when a loop-carried dependency has distance > 1:

```cpp
// AssignLatencies.cpp line 32
if (loopHasDistGreaterThanOne(forOp)) return false;

// PipelineExpander.cpp line 227-230
if (distance > 1) {
  LDBG("--only support loop carried dependency with a distance of 1 ... -> BAIL");
  return false;
}
```

**What distance > 1 means:** NOT about pipelining two separate loops. It means within a single loop, a value produced in iteration k is used in iteration k+2 or later (not just k+1). This creates multi-iteration live ranges that the current prologue/epilogue generation cannot handle.

**Why it matters:** CUTLASS's persistent kernels emit long prologues (prefetching multiple tiles before entering steady state). If Triton can't handle distance > 1, it can't generate these deep prologues for patterns where a load result is consumed N iterations later.

**Difficulty:** Medium-high. Requires:
1. Prologue length = `distance * II` instead of `II`
2. Buffer count = `distance + 1` instead of `2`
3. PipelineExpander must track multi-iteration live ranges
4. Well-understood in modulo scheduling literature, but engineering effort is substantial

---

## Problem 3: `scf.if` sink in non-WS loops (extra buffer needed)

**Status: Known limitation, HACK workaround in place**

In non-WS loops, Triton's PipelineExpander "sinks" `scf.if` operations to the end of the loop body rather than peeling them into the epilogue. This means the `if` condition and its results stay alive across the entire iteration, requiring an extra buffer for the live range overlap.

```cpp
// AssignLatencies.cpp lines 190-196
// HACK: A pipelined MMA's latency should equal the number of buffers
// for the accumulator, but when the user is in an `scf.if` in SWP,
// the `scf.if` is pushed to the end of the loop rather than peeled
// before the MMA op, requiring an extra buffer due to liverange
// overlap.
```

**Why it matters:** Extra buffer = more SMEM = fewer pipeline stages = lower overlap = worse performance. In WS mode this doesn't happen because MMA and epilogue are in separate partitions.

**Is this the same as the epilogue subtiling problem?** Partially related. Epilogue subtiling in CUTLASS reduces SMEM pressure by processing the output tile in smaller chunks, freeing SMEM for more pipeline stages. The `scf.if` sink problem is about unnecessary buffer allocation in the pipeline. Fixing the sink would free one buffer, which could then be used for an additional pipeline stage — essentially the same goal (more stages = better overlap).

---

## Problem 4: WS epilogue peeling not enabled

**Status: TODO in code, user tested and found no benefit**

```cpp
// SoftwarePipeliner.cpp line 116
// TODO: Enable epilogue peeling for warp specialized loops
```

Currently only MMAv5 (Blackwell) loops get `customEpiloguePeeling`. WS loops don't peel their epilogue, meaning the epilogue operations stay inside the loop body and can't overlap with the next tile's prologue.

User tested this and found no measurable benefit. This is likely because the bottleneck is deeper pipeline overlap (persistent kernel), not just epilogue peeling alone.

---

## Problem 5: Persistent kernel + epilogue subtiling (the real bottleneck)

**Status: Works in tutorial, not yet in production Triton**

From our benchmark on B200 (M=N=8192, K sweep):

| Kernel type | K=128 | K=2688 | K=7808 |
|---|---|---|---|
| TMA+WS non-persistent | 44% | 85% | 88% |
| TMA persistent+WS | 96% | 95% | 92% |
| cuBLAS | 100% | 100% | 100% |

**The gap between non-persistent and persistent is the main performance bottleneck.** Persistent kernels keep CTAs resident across tiles, amortizing launch overhead and maintaining steady-state pipeline rhythm. Epilogue subtiling reduces SMEM pressure to allow more pipeline stages.

This is the highest-impact direction: making Triton's persistent matmul kernel (from `09-persistent-matmul.py`) production-ready and generally available, not just a tutorial example.

---

## Priority Summary

| Priority | Problem | Impact | Difficulty |
|---|---|---|---|
| **1 (Highest)** | Persistent kernel + epilogue subtiling | 48% → 92% cuBLAS | Medium (tutorial exists) |
| **2** | `setmaxnreg` not emitted (minWarps=4 fix) | Unknown, potentially helps occupancy | Low (code change done) |
| **3** | SWP distance > 1 support | Enables deep prologues like CUTLASS | High |
| **4** | `scf.if` sink → extra buffer | Frees 1 buffer for more stages | Medium |
| **5** | WS epilogue peeling | User tested, no measurable benefit | Low (but niche) |