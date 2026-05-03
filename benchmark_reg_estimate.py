"""
A/B benchmark for register estimation optimization in optimizePartitionWarps.

Uses TMA + warp_specialize matmul kernel (required on Blackwell).
Outputs TFLOPS + IR info (requestedRegisters, partition warp counts).

Usage on B300:
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

from triton.tools.tensor_descriptor import TensorDescriptor


# ─── Warp-specialized TMA matmul kernel ───

@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_tma_ws_kernel(
    a_ptr, b_ptr, c_ptr,
    a_stride0, a_stride1,
    b_stride0, b_stride1,
    c_stride0, c_stride1,
    M, N, K,
    num_stages: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _compute_pid(pid, num_pid_n, num_pid_m, GROUP_M)

    k_tiles = tl.cdiv(K, BLOCK_K)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
        off_k = k * BLOCK_K
        a = a_desc.load((off_am, off_k))
        b = b_desc.load((off_bn, off_k))
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    c_desc.store((off_am, off_bn), c)


# ─── Persistent warp-specialized TMA matmul kernel ───

NUM_SMS = None  # Set at runtime

@triton.jit
def matmul_tma_ws_persistent_kernel(
    a_ptr, b_ptr, c_ptr,
    a_stride0, a_stride1,
    b_stride0, b_stride1,
    c_stride0, c_stride1,
    M, N, K,
    num_stages: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    NUM_SMS: tl.constexpr = 128,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    # Persistent: each CTA processes multiple tiles
    # Outer loop: persistent tile scheduling (no WS, just tile dispatch)
    # Inner loop: WS + pipelined K reduction (same as non-persistent)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
            off_k = k * BLOCK_K
            a = a_desc.load((off_am, off_k))
            b = b_desc.load((off_bn, off_k))
            accumulator = tl.dot(a, b.T, accumulator)

        c = accumulator.to(tl.float16)
        c_desc.store((off_am, off_bn), c)


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

    return sum(times) / len(times)


def run_triton_ws(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3):
    # TMA kernels need allocator for global scratch
    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)  # [N, K] for b.T
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    def fn():
        matmul_tma_ws_kernel[grid](
            a, b, c,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            M, N, K,
            num_stages=num_stages,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    ms = benchmark_fn(fn)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


def run_triton_ws_persistent(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3):
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)  # [N, K] for b.T
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_tiles),)

    def fn():
        matmul_tma_ws_persistent_kernel[grid](
            a, b, c,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            M, N, K,
            num_stages=num_stages,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_SMS=num_sms,
        )

    ms = benchmark_fn(fn)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


def run_cublas(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    def fn():
        torch.matmul(a, b)

    ms = benchmark_fn(fn)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


# ─── IR dump ───

def dump_ir_info(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3):
    """Get ttgir info by running the kernel and inspecting kernel.asm."""
    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.randn((M, N), device='cuda', dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    kernel = matmul_tma_ws_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
        num_stages=num_stages,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    info = {}
    try:
        ttgir = kernel.asm["ttgir"]
        info["has_warp_specialize"] = "ttg.warp_specialize" in ttgir
        info["tc_gen5_mma_count"] = ttgir.count("ttng.tc_gen5_mma")
        import re
        req = re.search(r'requestedRegisters\s*=\s*array<i32:\s*(.*?)>', ttgir)
        if req:
            info["requested_registers"] = req.group(1)
        nw = re.findall(r'num_warps\((\d+)\)', ttgir)
        info["partition_num_warps"] = nw

        # Extract warp_specialize section for debugging
        ws_match = re.search(
            r'ttg\.warp_specialize(.*?)(?:\n  } \{|\n\s*\})', ttgir, re.DOTALL
        )
        if ws_match:
            info["warp_specialize_snippet"] = ws_match.group(0)[:2000]  # first 2000 chars
    except Exception as e:
        info["ir_error"] = str(e)

    # Also dump persistent kernel IR
    try:
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        p_grid = min(num_sms, num_tiles)
        triton.set_allocator(alloc_fn)
        p_kernel = matmul_tma_ws_persistent_kernel[p_grid](
            a, b, c,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            M, N, K,
            num_stages=num_stages,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_SMS=num_sms,
        )
        p_ttgir = p_kernel.asm["ttgir"]
        p_req = re.search(r'requestedRegisters\s*=\s*array<i32:\s*(.*?)>', p_ttgir)
        if p_req:
            info["persist_requested_registers"] = p_req.group(1)
        p_nw = re.findall(r'num_warps\((\d+)\)', p_ttgir)
        info["persist_partition_num_warps"] = p_nw
        info["persist_has_warp_specialize"] = "ttg.warp_specialize" in p_ttgir
    except Exception as e:
        info["persist_ir_error"] = str(e)[:100]

    return info


# ─── Shapes ───

SHAPES = [
    ("small_128",  128,  128, 128),
    ("medium_256", 256,  256, 256),
    ("medium_512", 512,  512, 512),
    ("large_1k",   1024, 1024, 1024),
    ("large_2k",   2048, 2048, 2048),
    ("large_4k",   4096, 4096, 4096),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_reg_estimate.json")
    parser.add_argument("--dump-ir", action="store_true")
    parser.add_argument("--shapes", nargs="*", help="Override shapes (MxNxK)")
    parser.add_argument("--debug-dir", help="Save full ttgir files to this directory")
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

        os.system("rm -rf ~/.triton/cache")

        # Triton TMA warp-specialized (non-persistent)
        triton_ms, triton_tflops = run_triton_ws(M, N, K)
        print(f"  Triton WS (non-persistent): {triton_ms:.3f} ms, {triton_tflops:.3f} TFLOPS")

        # Triton TMA warp-specialized persistent
        persist_ms = 0; persist_tflops = 0
        try:
            persist_ms, persist_tflops = run_triton_ws_persistent(M, N, K)
            print(f"  Triton WS (persistent):     {persist_ms:.3f} ms, {persist_tflops:.3f} TFLOPS")
        except Exception as e:
            print(f"  Triton WS (persistent):     ERROR ({str(e)[:80]})")

        # cuBLAS baseline
        cublas_tflops = 0
        if M >= 128 and N >= 128 and K >= 128:
            cublas_ms, cublas_tflops = run_cublas(M, N, K)
            ratio = triton_tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0
            print(f"  cuBLAS:    {cublas_ms:.3f} ms, {cublas_tflops:.3f} TFLOPS")
            print(f"  Triton/cuBLAS: {ratio:.1f}%")

        result = {
            "name": name, "M": M, "N": N, "K": K,
            "triton_ms": triton_ms, "triton_tflops": triton_tflops,
            "cublas_tflops": cublas_tflops,
            "ratio_pct": triton_tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0,
            "persist_ms": persist_ms, "persist_tflops": persist_tflops,
            "persist_ratio_pct": persist_tflops / cublas_tflops * 100 if cublas_tflops > 0 and persist_tflops > 0 else 0,
        }

        if args.dump_ir:
            ir_info = dump_ir_info(M, N, K)
            result.update(ir_info)
            if "has_warp_specialize" in ir_info:
                print(f"  warp_specialize: {ir_info['has_warp_specialize']}")
            if "requested_registers" in ir_info:
                print(f"  requestedRegisters: {ir_info['requested_registers']}")
            if "tc_gen5_mma_count" in ir_info:
                print(f"  tc_gen5_mma count: {ir_info['tc_gen5_mma_count']}")
            if "partition_num_warps" in ir_info:
                print(f"  partition_num_warps: {ir_info['partition_num_warps']}")
            if "persist_requested_registers" in ir_info:
                print(f"  persist_reqRegisters: {ir_info['persist_requested_registers']}")
            if "persist_partition_num_warps" in ir_info:
                print(f"  persist_partition_warps: {ir_info['persist_partition_num_warps']}")
            if "persist_has_warp_specialize" in ir_info:
                print(f"  persist_warp_specialize: {ir_info['persist_has_warp_specialize']}")
            if "warp_specialize_snippet" in ir_info:
                print(f"  ws snippet:\n{ir_info['warp_specialize_snippet'][:500]}")

            # Save full ttgir for offline analysis
            if args.debug_dir:
                os.makedirs(args.debug_dir, exist_ok=True)
                ttgir_path = os.path.join(args.debug_dir, f"ttgir_{name}.txt")
                # Re-compile to get ttgir (already cached)
                a = torch.randn((M, K), device='cuda', dtype=torch.float16)
                b = torch.randn((N, K), device='cuda', dtype=torch.float16)
                c = torch.randn((M, N), device='cuda', dtype=torch.float16)
                def alloc_fn(size, align, stream):
                    return torch.empty(size, dtype=torch.int8, device="cuda")
                triton.set_allocator(alloc_fn)
                grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
                k = matmul_tma_ws_kernel[grid](a, b, c,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1), M, N, K,
                    num_stages=3, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
                with open(ttgir_path, "w") as f:
                    f.write(k.asm["ttgir"])
                print(f"  ttgir saved to {ttgir_path}")

        results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()