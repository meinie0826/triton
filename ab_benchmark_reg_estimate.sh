#!/bin/bash
# A/B benchmark for register estimation optimization.
# 
# This script:
# 1. Compiles Triton with BEFORE code (origin/main)
# 2. Runs benchmark → saves results_before.json
# 3. Compiles Triton with AFTER code (reg-estimation branch)
# 4. Runs benchmark → saves results_after.json
# 5. Compares the two
#
# Usage on B300 server:
#   cd /path/to/triton
#   bash ab_benchmark_reg_estimate.sh
#
# The script assumes you manually build Triton (ninja + pip install -e).

set -e

TRITON_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$TRITON_DIR/ab_results"
BRANCH_BEFORE="origin/main"
BRANCH_AFTER="reg-estimation"

echo "=== A/B Benchmark for Register Estimation ==="
echo "Before: $BRANCH_BEFORE"
echo "After:  $BRANCH_AFTER"
echo "Results dir: $RESULTS_DIR"

mkdir -p "$RESULTS_DIR"

# ─── Helper: compile Triton ───
compile_triton() {
    echo "Compiling Triton..."
    cd "$TRITON_DIR"
    
    # Clean build dir for reliability
    BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
    echo "Build dir: $BUILD_DIR"
    
    # Configure if needed
    if [ ! -f "$BUILD_DIR/build.ninja" ]; then
        echo "Configuring cmake..."
        cd "$BUILD_DIR"
        cmake -G Ninja "$TRITON_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_PROJECTS="mlir" \
            -DMLIR_DIR="$(python3 -c 'import mlir; print(mlir.__path__[0])')" \
            2>&1 | tail -5
        cd "$TRITON_DIR"
    fi
    
    # Build
    echo "Building with ninja..."
    ninja -C "$BUILD_DIR" 2>&1 | tail -5
    
    # Install
    echo "Installing..."
    pip install -e . --no-build-isolation --no-deps 2>&1 | tail -5
    
    echo "Compilation done."
}

# ─── Helper: run benchmark ───
run_benchmark() {
    local label="$1"
    local output="$RESULTS_DIR/results_${label}.json"
    
    echo "Clearing Triton cache..."
    rm -rf ~/.triton/cache
    
    echo "Running benchmark for $label..."
    python3 "$TRITON_DIR/benchmark_reg_estimate.py" \
        --output "$output" \
        --dump-ir
    
    echo "Benchmark done. Results: $output"
}

# ─── Step 1: BEFORE (origin/main) ───
echo ""
echo "=== Step 1: Compiling BEFORE ($BRANCH_BEFORE) ==="
cd "$TRITON_DIR"
git checkout "$BRANCH_BEFORE"
compile_triton
run_benchmark "before"

# ─── Step 2: AFTER (reg-estimation) ───
echo ""
echo "=== Step 2: Compiling AFTER ($BRANCH_AFTER) ==="
cd "$TRITON_DIR"
git checkout "$BRANCH_AFTER"
compile_triton
run_benchmark "after"

# ─── Step 3: Compare ───
echo ""
echo "=== Step 3: Comparison ==="
python3 -c "
import json

before = json.load(open('$RESULTS_DIR/results_before.json'))
after = json.load(open('$RESULTS_DIR/results_after.json'))

print(f'{'Shape':<20} {'Before TFLOPS':>15} {'After TFLOPS':>15} {'Delta':>10} {'Delta%':>8}')
print('-' * 75)

total_delta = 0
for b, a in zip(before, after):
    name = b['name']
    bt = b.get('tflops', 0)
    at = a.get('tflops', 0)
    delta = at - bt
    pct = (delta / bt * 100) if bt > 0 else 0
    total_delta += delta
    print(f'{name:<20} {bt:>15.3f} {at:>15.3f} {delta:>10.3f} {pct:>7.1f}%')

avg_pct = (total_delta / sum(b.get('tflops', 0) for b in before) * 100) if any(b.get('tflops', 0) for b in before) else 0
print(f'')
print(f'Average delta: {avg_pct:.1f}%')

# Also print requestedRegisters comparison
print()
print('=== requestedRegisters comparison ===')
for b, a in zip(before, after):
    name = b['name']
    br = b.get('requested_registers', 'N/A')
    ar = a.get('requested_registers', 'N/A')
    bw = b.get('has_warp_specialize', 'N/A')
    aw = a.get('has_warp_specialize', 'N/A')
    print(f'{name:<20} before_regs={br:>30}  after_regs={ar:>30}')
    print(f'{name:<20} before_ws={bw:>30}  after_ws={aw:>30}')
"

echo ""
echo "Results saved in $RESULTS_DIR/"
echo "DONE."