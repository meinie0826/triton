"""
Controlled GEMM ablations with automatic result dumping.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None


RESULTS_DIR = Path(__file__).resolve().parent / "results"


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
    LOAD_CACHE_MODIFIER: tl.constexpr,
    STORE_CACHE_MODIFIER: tl.constexpr,
    LOAD_EVICTION_POLICY: tl.constexpr,
    STORE_EVICTION_POLICY: tl.constexpr,
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
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
            cache_modifier=LOAD_CACHE_MODIFIER,
            eviction_policy=LOAD_EVICTION_POLICY,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
            cache_modifier=LOAD_CACHE_MODIFIER,
            eviction_policy=LOAD_EVICTION_POLICY,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(
        c_ptrs,
        c,
        mask=c_mask,
        cache_modifier=STORE_CACHE_MODIFIER,
        eviction_policy=STORE_EVICTION_POLICY,
    )


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

    c = acc.to(tl.float16) if OUTPUT_FP16 else acc.to(tl.bfloat16)
    c_desc.store([offs_am, offs_bn], c)


def launch_no_tma(a, b, *, variant, args):
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
        USE_GROUPED_ORDER=variant.use_grouped_order,
        LOAD_CACHE_MODIFIER=variant.load_cache_modifier,
        STORE_CACHE_MODIFIER=variant.store_cache_modifier,
        LOAD_EVICTION_POLICY=variant.load_eviction_policy,
        STORE_EVICTION_POLICY=variant.store_eviction_policy,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    return c


def launch_tma(a, b_t, *, variant, args):
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
        USE_GROUPED_ORDER=variant.use_grouped_order,
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
    load_cache_modifier: str = ""
    store_cache_modifier: str = ""
    load_eviction_policy: str = ""
    store_eviction_policy: str = ""


def select_variants(compare: str, tma_available: bool):
    variants = [
        Variant("gemm_wo_tma_wo_l2opt", False, False),
        Variant("gemm_wo_tma_w_l2opt", False, True),
        Variant("gemm_w_tma_wo_l2opt", True, False),
        Variant("gemm_w_tma_w_l2opt", True, True),
    ]
    if compare == "tma":
        variants = [v for v in variants if not v.use_grouped_order]
    elif compare == "l2":
        variants = [v for v in variants if not v.use_tma]
    elif compare == "cache":
        variants = [
            Variant("gemm_cache_none", False, False),
            Variant("gemm_cache_load_cg", False, False, load_cache_modifier=".cg"),
            Variant("gemm_cache_store_wt", False, False, store_cache_modifier=".wt"),
            Variant("gemm_cache_load_cg_store_wt", False, False, load_cache_modifier=".cg", store_cache_modifier=".wt"),
        ]
    elif compare == "eviction":
        variants = [
            Variant("gemm_eviction_none", False, False),
            Variant("gemm_eviction_load_last", False, False, load_eviction_policy="evict_last"),
            Variant("gemm_eviction_store_first", False, False, store_eviction_policy="evict_first"),
            Variant(
                "gemm_eviction_load_last_store_first",
                False,
                False,
                load_eviction_policy="evict_last",
                store_eviction_policy="evict_first",
            ),
        ]
    if not tma_available:
        variants = [v for v in variants if not v.use_tma]
    return variants


def run_variant(variant, a, b, b_t, ref, args):
    fn = (lambda: launch_tma(a, b_t, variant=variant, args=args)) if variant.use_tma else (
        lambda: launch_no_tma(a, b, variant=variant, args=args)
    )
    out = fn()
    torch.cuda.synchronize()
    max_abs = (out - ref).abs().max().item()
    mean_abs = (out - ref).abs().mean().item()
    latency_ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    tflops = 2 * args.M * args.N * args.K * 1e-12 / (latency_ms * 1e-3)
    return {
        "variant": variant.name,
        "use_tma": variant.use_tma,
        "use_l2opt": variant.use_grouped_order,
        "load_cache": variant.load_cache_modifier or '""',
        "store_cache": variant.store_cache_modifier or '""',
        "load_eviction": variant.load_eviction_policy or '""',
        "store_eviction": variant.store_eviction_policy or '""',
        "latency_ms": latency_ms,
        "tflops": tflops,
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
    }


def default_output_paths(compare: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{compare}"
    return RESULTS_DIR / f"{stem}.csv", RESULTS_DIR / f"{stem}.json"


def write_outputs(rows, meta, csv_path: Path, json_path: Path):
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--block-m", type=int, default=128)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--block-k", type=int, default=64)
    p.add_argument("--group-m", type=int, default=8)
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", choices=["all", "tma", "l2", "cache", "eviction"], default="all")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device is required"
    assert is_cuda(), "This benchmark currently targets CUDA"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    a = torch.randn((args.M, args.K), device="cuda", dtype=dtype)
    b = torch.randn((args.K, args.N), device="cuda", dtype=dtype)
    b_t = b.T.contiguous() if supports_tma() else None
    ref = torch.matmul(a, b)
    variants = select_variants(args.compare, supports_tma())
    rows = [run_variant(v, a, b, b_t, ref, args) for v in variants]

    baseline = rows[0]["latency_ms"]
    base_no_tma = next((r["latency_ms"] for r in rows if r["variant"] == "gemm_wo_tma_wo_l2opt"), None)
    base_no_l2 = base_no_tma
    for row in rows:
        row["speedup_vs_baseline"] = f"{baseline / row['latency_ms']:.4f}x"
        row["speedup_vs_wo_tma"] = ""
        row["speedup_vs_wo_l2opt"] = ""
        if row["use_tma"] and not row["use_l2opt"] and base_no_tma is not None:
            row["speedup_vs_wo_tma"] = f"{base_no_tma / row['latency_ms']:.4f}x"
        if row["use_l2opt"] and not row["use_tma"] and base_no_l2 is not None:
            row["speedup_vs_wo_l2opt"] = f"{base_no_l2 / row['latency_ms']:.4f}x"

    csv_path, json_path = args.csv, args.json
    if csv_path is None or json_path is None:
        auto_csv, auto_json = default_output_paths(args.compare)
        csv_path = csv_path or auto_csv
        json_path = json_path or auto_json

    meta = {
        "device": torch.cuda.get_device_name(),
        "supports_tma": supports_tma(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_outputs(rows, meta, csv_path, json_path)

    print(f"# csv={csv_path}")
    print(f"# json={json_path}")
    print("variant, use_tma, use_l2opt, load_cache, store_cache, load_eviction, store_eviction, latency_ms, tflops, max_abs_err, mean_abs_err, speedup_vs_baseline, speedup_vs_wo_tma, speedup_vs_wo_l2opt")
    for row in rows:
        print(
            f"{row['variant']}, {row['use_tma']}, {row['use_l2opt']}, {row['load_cache']}, {row['store_cache']}, "
            f"{row['load_eviction']}, {row['store_eviction']}, {row['latency_ms']:.6f}, {row['tflops']:.3f}, "
            f"{row['max_abs_err']:.6f}, {row['mean_abs_err']:.6f}, {row['speedup_vs_baseline']}, "
            f"{row['speedup_vs_wo_tma']}, {row['speedup_vs_wo_l2opt']}"
        )


if __name__ == "__main__":
    main()
