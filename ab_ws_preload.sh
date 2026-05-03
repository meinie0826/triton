#!/bin/bash
# A/B test: ws_preload vs ws_standard on current branch
# This tests whether hoisting the first TMA load before WS helps.
# No branch switching needed — both variants are in the same kernel file.

set -e

TRITON_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="/tmp/bench_ws_preload.py"

echo "=== WS Preload A/B Test ==="
echo "Triton dir: $TRITON_DIR"

cp "$TRITON_DIR/bench_ws_preload.py" "$BENCH_SCRIPT"

# Build current branch
echo ""
echo ">>> Building..."
cd "$TRITON_DIR"
make -j$(nproc)

echo ">>> Running benchmark..."
python3 "$BENCH_SCRIPT" current

echo ""
echo "=== Summary ==="
python3 << 'PYEOF'
import json

data = json.load(open("bench_ws_preload_current.json"))

# Group by shape+config, compare ws_standard vs ws_preload
from collections import defaultdict
groups = defaultdict(dict)
for r in data:
    key = f"{r['shape']}|{r['config']}"
    groups[key][r['variant']] = r

print(f"{'shape+config':30s} | {'std TF':>8s} | {'preload TF':>8s} | {'Δ%':>7s} | {'std%':>6s} | {'pre%':>6s} | {'persist TF':>10s} | {'persist+sub TF':>10s}")
print("-" * 110)

deltas_std_preload = []
deltas_persist = []
for key in sorted(groups.keys()):
    g = groups[key]
    std = g.get('ws_standard')
    pre = g.get('ws_preload')
    persist = g.get('ws_persistent')
    persist_sub = g.get('ws_persistent_subtile')
    
    line = f"{key:30s}"
    
    if std and std['tflops'] > 0:
        line += f" | {std['tflops']:8.2f}"
    else:
        line += f" | {'N/A':>8s}"
    
    if pre and pre['tflops'] > 0:
        line += f" | {pre['tflops']:8.2f}"
    else:
        line += f" | {'N/A':>8s}"
    
    if std and pre and std['tflops'] > 0 and pre['tflops'] > 0:
        delta = (pre['tflops'] - std['tflops']) / std['tflops'] * 100
        deltas_std_preload.append(delta)
        line += f" | {delta:+7.2f}%"
    else:
        line += f" | {'N/A':>7s}"
    
    if std and std['tflops'] > 0:
        line += f" | {std['ratio_pct']:6.1f}%"
    else:
        line += f" | {'N/A':>6s}"
    
    if pre and pre['tflops'] > 0:
        line += f" | {pre['ratio_pct']:6.1f}%"
    else:
        line += f" | {'N/A':>6s}"

    if persist and persist['tflops'] > 0:
        line += f" | {persist['tflops']:10.2f}"
    else:
        line += f" | {'N/A':>10s}"
    
    if persist_sub and persist_sub['tflops'] > 0:
        line += f" | {persist_sub['tflops']:10.2f}"
    else:
        line += f" | {'N/A':>10s}"

    print(line)

if deltas_std_preload:
    avg = sum(deltas_std_preload) / len(deltas_std_preload)
    print(f"\nAverage preload vs standard delta: {avg:+.2f}%")
PYEOF