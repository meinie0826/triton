# WS Prologue First-Load Overlap: Analysis & Improvement Plan

## Current Problem

In Triton's WS kernel on Blackwell, the timeline is:

```
default region
  → warp_specialize
    → partition0 (MMA, 1 warp): wait_barrier → MMA → MMA → ...
    → partition1 (TMA, 2 warps): TMA_load → signal → TMA_load → signal → ...
```

The first `wait_barrier` in partition0 means the MMA warp group sits idle while
the TMA warp group loads the first tile. This is **wasted time proportional to
one TMA load latency** (~2-3µs on B200).

## Why Prologue Peeling Didn't Help

The `epilogue-bench` branch tested three approaches:

| Approach | Avg Δ vs before | Notes |
|---|---|---|
| Epilogue peeling (standard) | -3~21% | Extra scf.if + code bloat in partition |
| Prologue-only peeling | -1~5% | Slightly better, still no gain |
| Prologue+epilogue peeling | -1~8% | Worst of both |

All approaches are negative because:
1. Prologue peeling clones the entire loop body into an `scf.if` before the for loop
2. The cloned MMA op still runs **serially** inside the same partition
3. No overlap is gained — MMA still waits for data before it can run
4. Extra code increases register pressure in the already-tiny 1-warp partition

The `tc_gen5_mma count` went from 1 → 2, confirming the prologue MMA is just
an extra serial op, not overlapped with anything.

## CUTLASS's Approach: First Load Before WS

CUTLASS SM90/SM100 persistent kernels do this:

```
kernel entry:
  first TMA load (by producer warp group)    ← BEFORE warp specialization
  signal barrier
warp specialization:
  MMA warp group: NO wait_barrier needed for first tile
  TMA warp group: second load → signal → third load → ...
```

The first load is issued **before** the MMA loop starts, so the MMA warp group
enters the loop with data already available. This eliminates the first
`wait_barrier` stall.

## Proposed Fix for Triton

The idea: hoist the first TMA load + barrier signal **out of** the WS loop
into the default region (before `ttg.warp_specialize`).

### Before (current):

```
ttg.warp_specialize {
  default { warp_yield }
  partition0 (MMA) {
    for k in range(k_tiles):
      wait_barrier   ← stall on first iteration
      MMA(...)
      commit_barrier
    warp_return
  }
  partition1 (TMA) {
    for k in range(k_tiles):
      TMA_load(A, k)      ← first load happens here
      TMA_load(B, k)
      signal_barrier
    warp_return
  }
}
```

### After (proposed):

```
# In default region, before warp_specialize:
TMA_load(A, 0)       ← hoisted first load
TMA_load(B, 0)
signal_barrier       ← data ready before WS starts

ttg.warp_specialize {
  default { warp_yield }
  partition0 (MMA) {
    # First iteration: data already available, no wait needed
    # OR: wait_barrier still present but immediately satisfied
    for k in range(k_tiles):
      if k > 0: wait_barrier   ← skip on first iteration
      MMA(...)
      commit_barrier
    warp_return
  }
  partition1 (TMA) {
    for k in range(k_tiles):
      if k > 0: TMA_load(A, k)  ← skip first, already done
      if k > 0: TMA_load(B, k)
      signal_barrier
    warp_return
  }
}
```

### Implementation Challenges

1. **TMA descriptor availability**: TMA descriptors (`tl.make_tensor_descriptor`)
   are created in the default region before WS, so they should be accessible
   for hoisting.

2. **Barrier indexing**: The multi-buffer barrier system uses index arithmetic
   (`memdesc_index`) to track which buffer is ready. Hoisting the first load
   means the barrier indices start at a different state when entering the loop.

3. **WS partition arguments**: The loaded data (SMEM buffers) and barrier tokens
   need to be passed into the WS partitions as arguments from the default region.

4. **Loop bounds**: If the first load is hoisted, the TMA partition loop should
   start from `k=1` instead of `k=0`. This requires adjusting `lowerBound`.

5. **num_stages interaction**: The current `num_stages=3` means 3 buffers for
   A and B. After hoisting, buffer 0 is already consumed, so the effective
   pipeline depth stays the same but the first stall is removed.

### Expected Impact

For a kernel with K=512 (8 iterations), removing one wait_barrier stall saves
approximately one TMA load latency (~2-3µs). For small K (few iterations), this
is a significant fraction of total runtime. For large K, the steady-state
pipeline dominates and the impact is smaller.

Based on the benchmark data, the biggest relative gains would be for:
- `k4iter_bk64` (K=256, 4 iterations): ~3-4% potential
- `k8iter_bk64` (K=512, 8 iterations): ~2-3% potential
- `medium_512` (512³): ~1-2% potential

For large shapes (4k³+), the gain would be <0.5% since steady-state dominates.

### Relationship to Other Issues

- **This is NOT the same as scf.if sink problem**: The sink problem is about
  unnecessary buffer allocation in the pipeline. This is about eliminating the
  first wait_barrier stall by hoisting a load before the loop.

- **This IS related to distance>1 support**: If we could properly support
  distance>1 in the SWP, the "first load before loop" pattern would be the
  simplest case (distance=K, where K is the loop trip count). However, since
  we're doing this at the WS/IR level rather than the SWP level, we can
  implement it without needing full distance>1 support.

- **This complements persistent kernel work**: Even with persistent kernels,
  each new tile still has a first-load stall. Hoisting eliminates that stall
  for every tile, not just the first one.

## Next Steps

1. Study how the WS partition gets its SMEM barrier arguments
2. Identify where to insert the first TMA load in the default region
3. Modify `PartitionBuilder` or `PartitionScheduling` to skip first iteration
   loads/wait_barrier
4. Test on B200/B300 with the same benchmark configs