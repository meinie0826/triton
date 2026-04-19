# GEMM Ablation Benchmarks

This directory contains controlled GEMM ablations for:

- `gemm_wo_tma` vs `gemm_w_tma`
- `gemm_wo_l2opt` vs `gemm_w_l2opt`
- `tl.load/tl.store` cache hint ablations
- `tl.load/tl.store` eviction policy ablations

## Build

From the repo root:

```bash
make
```

## Run

```bash
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare all --M 4096 --N 4096 --K 4096
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare tma --M 4096 --N 4096 --K 4096
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare l2 --M 4096 --N 4096 --K 4096
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare cache --M 4096 --N 4096 --K 4096
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare eviction --M 4096 --N 4096 --K 4096
```

Run a larger built-in shape sweep:

```bash
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare tma --shape-set large
```

Run a smaller, more IO-bound-oriented sweep:

```bash
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare tma --shape-set small
```

Run custom shapes:

```bash
PYTHONPATH=python python3 python/test/microbenchmark/gemm_ablation/run.py --compare tma --shapes 4096x4096x4096\;8192x8192x4096\;16384x8192x4096
```

Results are written to `python/test/microbenchmark/gemm_ablation/results/` by default.

## Notes

- TMA variants require CUDA plus tensor descriptor support.
- The TMA and non-TMA comparison uses the same non-transposed `B` layout `(K, N)` and the same `tl.dot(a, b)` formulation.
- Cache hint and eviction-policy ablations are currently wired to the pointer-based GEMM path.
- TMA stays in the benchmark for the main `tma` comparison, but Triton's current Python descriptor API does not expose cache hint knobs directly.
