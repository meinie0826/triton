#!/bin/bash
#=============================================================================
# Run the tutorial 09-persistent-matmul.py benchmark on B200/B300.
# This gives us the peak Triton performance reference to compare against.
#
# Usage on B300:
#   cd /workspace/triton
#   bash bench_tutorial.sh
#=============================================================================
set -euo pipefail

TRITON_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Tutorial 09 Persistent Matmul Benchmark ==="
echo ""

# Run the tutorial benchmark with FP16, K=512 to 4096
cd "$TRITON_DIR"
echo ">>> Running tutorial benchmark..."
python3 python/tutorials/09-persistent-matmul.py --prec fp16 --K_range 512 4096 --K_step 512

echo ""
echo "DONE."