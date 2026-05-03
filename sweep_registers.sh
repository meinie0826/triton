#!/bin/bash
#=============================================================================
# Register allocation sweep for persistent + WS matmul on Blackwell.
#
# Sweeps MMA partition requestedRegisters from 24 to 200+ while keeping
# TMA partition at 24. MMA partition forced to 4 warps (for setmaxnreg).
#
# Uses TRITON_REG_ESTIMATE_OVERRIDE env var to bypass the 40 magic number.
#
# Usage on B300:
#   cd /workspace/triton
#   bash sweep_registers.sh
#=============================================================================
set -euo pipefail

TRITON_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$TRITON_DIR/sweep_results"
BENCH_SCRIPT="/tmp/benchmark_reg_estimate.py"

mkdir -p "$RESULTS_DIR"

# ─── Save benchmark script ───
cp "$TRITON_DIR/benchmark_reg_estimate.py" "$BENCH_SCRIPT"

# ─── Build Triton (with env var override support) ───
echo ">>> Building Triton..."
cd "$TRITON_DIR"
make 2>&1 | tail -5
echo ">>> Build done."

# ─── Sweep values ───
# CUTLASS gives MMA ~208 regs/thread after dealloc. Test a wide range.
REG_VALUES=(24 32 40 48 56 64 72 80 96 120 152 200)

echo ""
echo "=== Register Sweep: MMA partition requestedRegisters ==="
echo "Testing values: ${REG_VALUES[*]}"
echo ""

for MMA_REG in "${REG_VALUES[@]}"; do
    # Override format: "MMA_REG,TMA_REG" for 2 partitions
    OVERRIDE="${MMA_REG},24"
    echo ">>> Testing requestedRegisters=${OVERRIDE}"
    rm -rf ~/.triton/cache

    export TRITON_REG_ESTIMATE_OVERRIDE="${OVERRIDE}"
    python3 "$BENCH_SCRIPT" \
        --output "$RESULTS_DIR/sweep_${MMA_REG}.json" \
        --dump-ir \
        --shapes "2048x2048x2048" "4096x4096x4096" "8192x8192x8192"
    unset TRITON_REG_ESTIMATE_OVERRIDE

    echo ">>> Done: sweep_${MMA_REG}.json"
    echo ""
done

# ─── Summary ───
echo "=========================================================="
echo "  Sweep Summary"
echo "=========================================================="
python3 -c "
import json, os
results_dir = '$RESULTS_DIR'
reg_values = [24, 32, 40, 48, 56, 64, 72, 80, 96, 120, 152, 200]
print(f'{\"MMA_Reg\":>8} {\"2k_nop TF\":>10} {\"2k_per TF\":>10} {\"4k_nop TF\":>10} {\"4k_per TF\":>10} {\"8k_nop TF\":>10} {\"8k_per TF\":>10} {\"2k/cuB%\":>8} {\"4k/cuB%\":>8} {\"8k/cuB%\":>8}')
print('-'*100)
for reg in reg_values:
    f = os.path.join(results_dir, f'sweep_{reg}.json')
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    # 2k = index 0, 4k = index 1, 8k = index 2 (from --shapes override)
    nop_2k = d[0].get('triton_tflops', 0)
    per_2k = d[0].get('persist_tflops', 0)
    nop_4k = d[1].get('triton_tflops', 0)
    per_4k = d[1].get('persist_tflops', 0)
    nop_8k = d[2].get('triton_tflops', 0)
    per_8k = d[2].get('persist_tflops', 0)
    cb_2k = d[0].get('cublas_tflops', 0)
    cb_4k = d[1].get('cublas_tflops', 0)
    cb_8k = d[2].get('cublas_tflops', 0)
    r2k = per_2k/cb_2k*100 if cb_2k else 0
    r4k = per_4k/cb_4k*100 if cb_4k else 0
    r8k = per_8k/cb_8k*100 if cb_8k else 0
    print(f'{reg:>8} {nop_2k:>10.1f} {per_2k:>10.1f} {nop_4k:>10.1f} {per_4k:>10.1f} {nop_8k:>10.1f} {per_8k:>10.1f} {r2k:>7.1f}% {r4k:>7.1f}% {r8k:>7.1f}%')
"

echo ""
echo "DONE. Results in $RESULTS_DIR/"