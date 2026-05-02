"""
A/B benchmark for register estimation optimization in optimizePartitionWarps.

Benchmarks warp-specialized matmul with:
  - Triton (warp_specialize=True) → our kernel
  - cuBLAS baseline (torch.matmul) → for reference

Outputs TFLOPS + IR info (partition warp counts, requestedRegisters).

Usage on B300:
  # After compiling Triton:
  python benchmark_reg_estimate.py --output results.json --dump-ir
"""

import argparse
import json
import os
import sys

import torch

import triton
import triton.language as tl

try:
    from triton._internal_testing import is_hip, is_blackwell
except ImportError:
    is_hip = False
    is_blackwell = True


# ─── Warp-specialized matmul kernel ───

@triton.jit
def matmul_ws_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(0, K, BLOCK_K, warp_specialize=True):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_iter, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_iter, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ─── Timing ───

def benchmark_fn(fn, n_warmup=10, n_repeat=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    return avg_ms


def run_triton_ws(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    def fn():
        c = torch.empty((M, N), device='cuda', dtype=torch.float16)
        matmul_ws_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    ms = benchmark_fn(fn)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


def run_cublas(M, N, K):
    """cuBLAS baseline via torch.matmul (fp16 input, fp16 output)."""
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    def fn():
        torch.matmul(a, b)

    ms = benchmark_fn(fn)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


# ─── IR dump ───

def dump_ir_info(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    # Run once to compile
    matmul_ws_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    info = {}
    try:
        ttgir = c.device_driver.get_kernel(
            matmul_ws_kernel[grid].fn,
            matmul_ws_kernel[grid].warmup
        ).asm["ttgir"]
        info["has_warp_specialize"] = "ttg.warp_specialize" in ttgir
        info["tc_gen5_mma_count"] = ttgir.count("ttng.tc_gen5_mma")

        import re
        req = re.search(r'requestedRegisters\s*=\s*array<i32:\s*(.*?)>', ttgir)
        if req:
            info["requested_registers"] = req.group(1)
        nw = re.findall(r'num_warps\((\d+)\)', ttgir)
        info["partition_num_warps"] = nw
    except Exception as e:
        # Try alternative access path
        try:
            import triton.compiler as tc
            kernel = matmul_ws_kernel[grid]
            # The kernel stores compiled metadata
            compiled_kernel = kernel._triton_kernel_
            ttgir = compiled_kernel.asm["ttgir"]
            info["has_warp_specialize"] = "ttg.warp_specialize" in ttgir
            info["tc_gen5_mma_count"] = ttgir.count("ttng.tc_gen5_mma")
            import re
            req = re.search(r'requestedRegisters\s*=\s*array<i32:\s*(.*?)>', ttgir)
            if req:
                info["requested_registers"] = req.group(1)
        except Exception as e2:
            info["ir_error"] = f"{e}; {e2}"

    return info


# ─── Shapes ───

SHAPES = [
    ("tiny_64",    64,   64,  64),
    ("small_128",  128,  128, 128),
    ("medium_256", 256,  256, 256),
    ("medium_512", 512,  512, 512),
    ("large_1k",   1024, 1024, 1024),
    ("large_2k",   2048, 2048, 2048),
    ("large_4k",   4096, 4096, 4096),
    # Non-square
    ("tall_256_64", 256,  64,  256),
    ("wide_64_256", 64,   256, 256),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_reg_estimate.json")
    parser.add_argument("--dump-ir", action="store_true")
    parser.add_argument("--shapes", nargs="*", help="Override shapes (MxNxK)")
    parser.add_argument("--skip-cublas", action="store_true", help="Skip cuBLAS baseline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device"); sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    shapes = SHAPES
    if args.shapes:
        shapes = [("custom", *map(int, s.split("x"))) for s in args.shapes]

    results = []
    for name, M, N, K in shapes:
        print(f"\n=== {name}: {M}x{N}x{K} ===")

        # Triton warp-specialized
        os.system("rm -rf ~/.triton/cache")
        triton_ms, triton_tflops = run_triton_ws(M, N, K)
        print(f"  Triton WS: {triton_ms:.3f} ms, {triton_tflops:.3f} TFLOPS")

        # cuBLAS
        cublas_tflops = 0
        if not args.skip_cublas and M >= 128 and N >= 128 and K >= 128:
            cublas_ms, cublas_tflops = run_cublas(M, N, K)
            ratio = triton_tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0
            print(f"  cuBLAS:    {cublas_ms:.3f} ms, {cublas_tflops:.3f} TFLOPS")
            print(f"  Triton/cuBLAS: {ratio:.1f}%")

        result = {
            "name": name, "M": M, "N": N, "K": K,
            "triton_ms": triton_ms, "triton_tflops": triton_tflops,
            "cublas_tflops": cublas_tflops,
            "ratio_pct": triton_tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0,
        }

        if args.dump_ir:
            os.system("rm -rf ~/.triton/cache")
            ir_info = dump_ir_info(M, N, K)
            result.update(ir_info)
            if "requested_registers" in ir_info:
                print(f"  requestedRegisters: {ir_info['requested_registers']}")

        results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()