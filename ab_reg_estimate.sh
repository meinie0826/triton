#!/bin/bash
#=============================================================================
# A/B benchmark for register estimation optimization.
#
# BEFORE = origin/main (no changes, requestedRegisters=24,24 on Blackwell)
# AFTER  = reg-estimation branch (with DotOpInterface detection)
#
# Usage on B300:
#   cd /workspace/triton
#   bash ab_reg_estimate.sh
#=============================================================================
set -euo pipefail

TRITON_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$TRITON_DIR/ab_results"
BENCH_SCRIPT="/tmp/benchmark_reg_estimate.py"
BRANCH_AFTER="reg-estimation"

mkdir -p "$RESULTS_DIR"

echo "=== Register Estimation A/B Benchmark ==="
echo ""

# ─── Save benchmark script to /tmp ───
if [ ! -f "$TRITON_DIR/benchmark_reg_estimate.py" ]; then
    echo "ERROR: benchmark_reg_estimate.py not found in $TRITON_DIR"
    exit 1
fi
cp "$TRITON_DIR/benchmark_reg_estimate.py" "$BENCH_SCRIPT"
echo "Benchmark script saved to $BENCH_SCRIPT"

# ─── Helper: build Triton ───
build_triton() {
    echo ">>> Building Triton (make)..."
    cd "$TRITON_DIR"
    make 2>&1 | tail -10
    echo ">>> Build done."
}

# ─── Helper: run benchmark ───
run_bench() {
    local label="$1"
    local output="$RESULTS_DIR/results_${label}.json"
    echo ">>> Clearing Triton cache..."
    rm -rf ~/.triton/cache
    echo ">>> Running benchmark: $label"
    python3 "$BENCH_SCRIPT" --output "$output" --dump-ir
    echo ">>> Done: $output"
}

# ─── Step 1: AFTER (current branch with changes) ───
echo "=========================================================="
echo "  STEP 1: AFTER (reg-estimation branch with DotOpInterface)"
echo "=========================================================="
cd "$TRITON_DIR"
git checkout "$BRANCH_AFTER"
build_triton
run_bench "after"

# ─── Step 2: BEFORE (origin/main, no changes) ───
echo ""
echo "=========================================================="
echo "  STEP 2: BEFORE (origin/main, baseline)"
echo "=========================================================="
cd "$TRITON_DIR"
git checkout origin/main
build_triton
run_bench "before"

# ─── Step 3: Switch back ───
cd "$TRITON_DIR"
git checkout "$BRANCH_AFTER"

# ─── Step 4: Compare ───
echo ""
echo "=========================================================="
echo "  STEP 3: Comparison"
echo "=========================================================="
python3 -c "
import json
b = json.load(open('$RESULTS_DIR/results_before.json'))
a = json.load(open('$RESULTS_DIR/results_after.json'))
print(f'{\"Shape\":<15} {\"Before TF\":>10} {\"After TF\":>10} {\"Delta%\":>8} {\"B/cuBL%\":>8} {\"A/cuBL%\":>8} {\"reqRegs_B\":>16} {\"reqRegs_A\":>16}')
print('-'*110)
for bi, ai in zip(b, a):
    n = bi['name']
    bt, at = bi['triton_tflops'], ai['triton_tflops']
    d = (at-bt)/bt*100 if bt else 0
    bc = bi.get('ratio_pct', 0)
    ac = ai.get('ratio_pct', 0)
    br = bi.get('requested_registers','N/A')
    ar = ai.get('requested_registers','N/A')
    print(f'{n:<15} {bt:>10.2f} {at:>10.2f} {d:>7.1f}% {bc:>7.1f}% {ac:>7.1f}% {str(br):>16} {str(ar):>16}')
ds = [ai['triton_tflops']-bi['triton_tflops'] for bi,ai in zip(b,a)]
avg = sum(d/bi['triton_tflops'] for d,bi in zip(ds,b))/len(ds)*100
print(f'Average delta: {avg:+.1f}%')
"
echo ""
echo "DONE. Results in $RESULTS_DIR/"