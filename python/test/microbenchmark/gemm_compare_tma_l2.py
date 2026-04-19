"""
Controlled GEMM benchmark for TMA and L2-ordering comparisons.

This script keeps the tile shape and launch parameters fixed, then compares:
1. gemm_wo_tma vs gemm_w_tma
2. gemm_wo_l2opt vs gemm_w_l2opt

Examples:
  python python/test/microbenchmark/gemm_compare_tma_l2.py --M 8192 --N 8192 --K 8192
  python python/test/microbenchmark/gemm_compare_tma_l2.py --compare tma --M 4096 --N 8192 --K 8192
  python python/test/microbenchmark/gemm_compare_tma_l2.py --compare l2 --M 4096 --N 8192 --K 8192 --csv /tmp/gemm.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch

import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None


def is_cuda() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma() -> bool:
    return is_cuda() and TensorDescriptor is not None and torch.cuda.get_device_capability()[0] >= 9


@triton.jit
def _compute_pid(tile_id, num_pid_m, num_pid_n, group_size_m, use_grouped_order: tl.constexpr):
    if use_grouped_order:
        num_pid_in_group = group_size_m * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * group_size_m
        actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % actual_group_size_m)
        pid_n = (tile_id % num_pid_in_group) // actual_group_size_m
    else:
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
    return pid_m, pid_n


@triton.jit
def matmul_kernel_no_tma(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_GROUPED_ORDER: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M, USE_GROUPED_ORDER)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def matmul_kernel_tma(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_GROUPED_ORDER: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M, USE_GROUPED_ORDER)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        acc = tl.dot(a, b.T, acc)

    if OUTPUT_FP16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.bfloat16)
    c_desc.store([offs_am, offs_bn], c)


def launch_no_tma(a: torch.Tensor, b: torch.Tensor, *, use_grouped_order: bool, args) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, args.block_m) * triton.cdiv(n, args.block_n),)
    matmul_kernel_no_tma[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=args.block_m,
        BLOCK_SIZE_N=args.block_n,
        BLOCK_SIZE_K=args.block_k,
        GROUP_SIZE_M=args.group_m,
        USE_GROUPED_ORDER=use_grouped_order,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    return c


def launch_tma(a: torch.Tensor, b_t: torch.Tensor, *, use_grouped_order: bool, args) -> torch.Tensor:
    m, k = a.shape
    n, _ = b_t.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    a_desc = TensorDescriptor.from_tensor(a, [1, 1])
    b_desc = TensorDescriptor.from_tensor(b_t, [1, 1])
    c_desc = TensorDescriptor.from_tensor(c, [1, 1])
    a_desc.block_shape = [args.block_m, args.block_k]
    b_desc.block_shape = [args.block_n, args.block_k]
    c_desc.block_shape = [args.block_m, args.block_n]

    grid = (triton.cdiv(m, args.block_m) * triton.cdiv(n, args.block_n),)
    matmul_kernel_tma[grid](
        a_desc,
        b_desc,
        c_desc,
        m,
        n,
        k,
        BLOCK_SIZE_M=args.block_m,
        BLOCK_SIZE_N=args.block_n,
        BLOCK_SIZE_K=args.block_k,
        GROUP_SIZE_M=args.group_m,
        USE_GROUPED_ORDER=use_grouped_order,
        OUTPUT_FP16=a.dtype == torch.float16,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    return c


@dataclass(frozen=True)
class Variant:
    name: str
    use_tma: bool
    use_grouped_order: bool


def benchmark_variant(variant: Variant, a: torch.Tensor, b: torch.Tensor, b_t: torch.Tensor | None, args):
    if variant.use_tma:
        assert b_t is not None
        fn = lambda: launch_tma(a, b_t, use_grouped_order=variant.use_grouped_order, args=args)
    else:
        fn = lambda: launch_no_tma(a, b, use_grouped_order=variant.use_grouped_order, args=args)
    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    tflops = 2 * args.M * args.N * args.K * 1e-12 / (ms * 1e-3)
    return ms, tflops


def validate_variant(variant: Variant, a: torch.Tensor, b: torch.Tensor, b_t: torch.Tensor | None, ref: torch.Tensor, args):
    if variant.use_tma:
        out = launch_tma(a, b_t, use_grouped_order=variant.use_grouped_order, args=args)
    else:
        out = launch_no_tma(a, b, use_grouped_order=variant.use_grouped_order, args=args)
    torch.cuda.synchronize()
    max_abs = (out - ref).abs().max().item()
    mean_abs = (out - ref).abs().mean().item()
    return max_abs, mean_abs


def select_variants(compare: str, tma_available: bool) -> list[Variant]:
    all_variants = [
        Variant("gemm_wo_tma_wo_l2opt", use_tma=False, use_grouped_order=False),
        Variant("gemm_wo_tma_w_l2opt", use_tma=False, use_grouped_order=True),
        Variant("gemm_w_tma_wo_l2opt", use_tma=True, use_grouped_order=False),
        Variant("gemm_w_tma_w_l2opt", use_tma=True, use_grouped_order=True),
    ]
    if compare == "tma":
        all_variants = [v for v in all_variants if not v.use_grouped_order]
    elif compare == "l2":
        all_variants = [v for v in all_variants if not v.use_tma]
    if not tma_available:
        all_variants = [v for v in all_variants if not v.use_tma]
    return all_variants


def print_results(rows: list[dict[str, object]], compare: str):
    print(
        "variant, use_tma, use_l2opt, latency_ms, tflops, max_abs_err, mean_abs_err, "
        "speedup_vs_wo_tma, speedup_vs_wo_l2opt"
    )
    for row in rows:
        print(
            f"{row['variant']}, {row['use_tma']}, {row['use_l2opt']}, "
            f"{row['latency_ms']:.6f}, {row['tflops']:.3f}, {row['max_abs_err']:.6f}, "
            f"{row['mean_abs_err']:.6f}, {row['speedup_vs_wo_tma']}, {row['speedup_vs_wo_l2opt']}"
        )
    if compare == "all":
        print("\nsummary:")
        tma_rows = [r for r in rows if not r["use_l2opt"]]
        l2_rows = [r for r in rows if not r["use_tma"]]
        if len(tma_rows) == 2:
            speedup = tma_rows[0]["latency_ms"] / tma_rows[1]["latency_ms"]
            print(f"gemm_w_tma vs gemm_wo_tma speedup: {speedup:.4f}x")
        if len(l2_rows) == 2:
            speedup = l2_rows[0]["latency_ms"] / l2_rows[1]["latency_ms"]
            print(f"gemm_w_l2opt vs gemm_wo_l2opt speedup: {speedup:.4f}x")


def maybe_write_csv(rows: list[dict[str, object]], csv_path: str | None):
    if csv_path is None:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "use_tma",
                "use_l2opt",
                "latency_ms",
                "tflops",
                "max_abs_err",
                "mean_abs_err",
                "speedup_vs_wo_tma",
                "speedup_vs_wo_l2opt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--group-m", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", choices=["all", "tma", "l2"], default="all")
    parser.add_argument("--csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device is required"
    assert is_cuda(), "This benchmark currently targets the CUDA backend"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    a = torch.randn((args.M, args.K), device=device, dtype=dtype)
    b = torch.randn((args.K, args.N), device=device, dtype=dtype)
    b_t = b.T.contiguous() if supports_tma() else None
    ref = torch.matmul(a, b)

    variants = select_variants(args.compare, supports_tma())
    if not variants:
        raise RuntimeError("No benchmark variants selected")

    rows: list[dict[str, object]] = []
    baseline_no_tma_ms = None
    baseline_no_l2_ms = None

    for variant in variants:
        max_abs_err, mean_abs_err = validate_variant(variant, a, b, b_t, ref, args)
        latency_ms, tflops = benchmark_variant(variant, a, b, b_t, args)
        row = {
            "variant": variant.name,
            "use_tma": variant.use_tma,
            "use_l2opt": variant.use_grouped_order,
            "latency_ms": latency_ms,
            "tflops": tflops,
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
            "speedup_vs_wo_tma": "",
            "speedup_vs_wo_l2opt": "",
        }
        rows.append(row)
        if variant.name == "gemm_wo_tma_wo_l2opt":
            baseline_no_tma_ms = latency_ms
            baseline_no_l2_ms = latency_ms

    for row in rows:
        if row["use_l2opt"] is False and baseline_no_tma_ms is not None and row["use_tma"]:
            row["speedup_vs_wo_tma"] = f"{baseline_no_tma_ms / row['latency_ms']:.4f}x"
        if row["use_tma"] is False and baseline_no_l2_ms is not None and row["use_l2opt"]:
            row["speedup_vs_wo_l2opt"] = f"{baseline_no_l2_ms / row['latency_ms']:.4f}x"

    print(
        f"# device={torch.cuda.get_device_name()} M={args.M} N={args.N} K={args.K} "
        f"dtype={args.dtype} block=({args.block_m},{args.block_n},{args.block_k}) "
        f"group_m={args.group_m} num_warps={args.num_warps} num_stages={args.num_stages}"
    )
    if not supports_tma():
        print("# TMA is not available on this GPU/runtime, so TMA variants were skipped.")
    print_results(rows, args.compare)
    maybe_write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
