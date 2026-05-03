#!/bin/bash
# A/B test script for Triton matmul benchmarks
# BEFORE: origin/main (baseline)
# AFTER: current branch (with optimizations)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_SCRIPT="/tmp/bench_comprehensive.py"
RESULTS_DIR="$TRITON_DIR/ab_results"

echo "=== Triton Matmul A/B Test ==="
echo "Triton dir: $TRITON_DIR"

# Copy benchmark script to /tmp ( survives git checkout )
cp "$SCRIPT_DIR/bench_comprehensive.py" "$BENCH_SCRIPT"

mkdir -p "$RESULTS_DIR"

# ─── AFTER (current branch) ───
CURRENT_BRANCH=$(git -C "$TRITON_DIR" rev-parse --abbrev-ref HEAD)
echo ""
echo ">>> Building AFTER ($CURRENT_BRANCH)..."
cd "$TRITON_DIR"
make -j$(nproc)
echo ">>> Running AFTER benchmark..."
python3 "$BENCH_SCRIPT" after --dump-ir
mv bench_comprehensive_after.json "$RESULTS_DIR/results_after.json"

# ─── BEFORE (origin/main) ───
echo ""
echo ">>> Switching to origin/main..."
cd "$TRITON_DIR"
git checkout origin/main
echo ">>> Building BEFORE (origin/main)..."
make -j$(nproc)
echo ">>> Running BEFORE benchmark..."
python3 "$BENCH_SCRIPT" before --dump-ir
mv bench_comprehensive_before.json "$RESULTS_DIR/results_before.json"

# ─── Switch back ───
echo ""
echo ">>> Switching back to $CURRENT_BRANCH..."
cd "$TRITON_DIR"
git checkout "$CURRENT_BRANCH"

# ─── Compare ───
echo ""
echo "=== Comparison Results ==="
python3 << 'PYEOF'
import json, sys

before = json.load(open("ab_results/results_before.json"))
after = json.load(open("ab_results/results_after.json"))

# Group by shape+config+variant
b_dict = {f"{r['shape']}|{r['config']}|{r['variant']}": r for r in before}
a_dict = {f"{r['shape']}|{r['config']}|{r['variant']}": r for r in after}

keys = sorted(set(b_dict.keys()) & set(a_dict.keys()))

print(f"{'key':50s} | {'before':>10s} | {'after':>10s} | {'Δ%':>8s} | {'ratio_b':>8s} | {'ratio_a':>8s}")
print("-" * 100)

deltas = []
for k in keys:
    b = b_dict[k]
    a = a_dict[k]
    if b['tflops'] <= 0 or a['tflops'] <= 0:
        continue
    delta = (a['tflops'] - b['tflops']) / b['tflops'] * 100
    deltas.append(delta)
    print(f"{k:50s} | {b['tflops']:10.3f} | {a['tflops']:10.3f} | {delta:8.2f}% | {b['ratio_pct']:8.1f}% | {a['ratio_pct']:8.1f}%")

if deltas:
    avg_delta = sum(deltas) / len(deltas)
    print(f"\nAverage TFLOPS delta: {avg_delta:.2f}%")

# Print IR info for WS variants
print("\n=== IR Info ===")
for k in keys:
    a = a_dict[k]
    if a.get('variant') == 'tma_ws':
        ir_keys = ['requested_registers', 'partition_num_warps', 'setmaxnreg_count']
        info_str = " | ".join(f"{ik}: {a.get(ik, 'N/A')}" for ik in ir_keys)
        print(f"  {k}: {info_str}")
PYEOF

echo ""
echo "Done! Results in $RESULTS_DIR/"