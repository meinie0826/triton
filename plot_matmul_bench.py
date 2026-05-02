#!/usr/bin/env python3
"""Plot Triton matmul benchmark results vs cuBLAS on B200."""

import json
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

data_path = sys.argv[1] if len(sys.argv) > 1 else "matmul_bench_data.json"
with open(data_path) as f:
    data = json.load(f)

Ks = [d["K"] for d in data["data"]]
kernels = ["naive", "tma", "tma_ws", "tma_persistent_ws"]
kernel_labels = [
    "naive (tl.load)",
    "TMA (no WS)",
    "TMA+WS (non-persistent)",
    "TMA persistent+WS",
]
colors = ["#4ECDC4", "#45B7D1", "#F7DC6F", "#E74C3C"]

# ─── Figure 1: Absolute TFLOPS/s ───
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(Ks))
width = 0.15

# cuBLAS baseline
cublas_vals = [d["cuBLAS"] for d in data["data"]]
axes[0].bar(x - 2*width, cublas_vals, width, label="cuBLAS", color="#2C3E50", edgecolor="white")

for i, (kernel, label, color) in enumerate(zip(kernels, kernel_labels, colors)):
    vals = [d[kernel] for d in data["data"]]
    offset = x + (i - 1) * width
    axes[0].bar(offset, vals, width, label=label, color=color, edgecolor="white")

axes[0].set_xlabel("K (GEMM_K dimension)", fontsize=12)
axes[0].set_ylabel("TFLOPS/s (fp16)", fontsize=12)
axes[0].set_title(f'Triton Matmul Absolute Performance\nM=N={data["M"]}, {data["dtype"]}, {data["device"]}', fontsize=13)
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(k) for k in Ks])
axes[0].legend(fontsize=9, loc="upper left")
axes[0].set_ylim(0, max(cublas_vals) * 1.15)
axes[0].grid(axis='y', alpha=0.3)

# ─── Figure 2: Ratio to cuBLAS (%) ───
for i, (kernel, label, color) in enumerate(zip(kernels, kernel_labels, colors)):
    ratios = [d[kernel] / d["cuBLAS"] * 100 for d in data["data"]]
    offset = x + (i - 1.5) * width
    axes[1].bar(offset, ratios, width, label=label, color=color, edgecolor="white")

axes[1].axhline(y=100, color="#2C3E50", linestyle="--", linewidth=1.5, label="cuBLAS baseline")
axes[1].set_xlabel("K (GEMM_K dimension)", fontsize=12)
axes[1].set_ylabel("% of cuBLAS", fontsize=12)
axes[1].set_title(f'Triton / cuBLAS Ratio\nM=N={data["M"]}, {data["dtype"]}, {data["device"]}', fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(k) for k in Ks])
axes[1].legend(fontsize=9, loc="lower right")
axes[1].set_ylim(30, 110)
axes[1].grid(axis='y', alpha=0.3)

# Add percentage labels on bars for the ratio chart
for i, (kernel, label, color) in enumerate(zip(kernels, kernel_labels, colors)):
    ratios = [d[kernel] / d["cuBLAS"] * 100 for d in data["data"]]
    offset = x + (i - 1.5) * width
    for j, (r, o) in enumerate(zip(ratios, offset)):
        axes[1].text(o, r + 1, f'{r:.0f}%', ha='center', va='bottom', fontsize=7, color=color)

plt.tight_layout()
out_path = data_path.replace(".json", "_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")