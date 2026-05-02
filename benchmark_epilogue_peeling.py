"""Benchmark warp-specialized epilogue peeling before/after.

Uses the matmul_tma_ws_kernel from test_warp_specialization.py
which has warp_specialize=True in the K loop.
"""
import json
import sys
import os
import torch
import triton
from triton._internal_testing import is_hip, is_blackwell
import triton.language as tl

if not is_hip() and torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None
    sys.exit("Requires CUDA + Blackwell GPU")


@triton.jit
def matmul_tma_ws_kernel(
        a_ptr, b_ptr, c_ptr,
        a_stride0, a_stride1,
        b_stride0, b_stride1,
        c_stride0, c_stride1,
        M, N, K,
        num_stages: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        USE_FP8: tl.constexpr,
        A_USE_TMA: tl.constexpr,
        B_USE_TMA: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    off_am = pid_m * BLOCK_SIZE_M
    off_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
        off_k = k * BLOCK_SIZE_K
        a = a_desc.load((off_am, off_k))
        b = b_desc.load((off_bn, off_k))
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float8e4nv if USE_FP8 else tl.float16)
    c_desc.store((off_am, off_bn), c)


def exceeds_smem_capacity(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, use_fp8):
    return (num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N) * (1 if use_fp8 else 2) > 228 * 1024


def alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def benchmark_shape(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps=4,
                    num_warmup=10, num_iters=50, use_fp8=False, a_tma=True, b_tma=True):
    """Run a single shape benchmark and return TFLOPS."""
    if exceeds_smem_capacity(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, use_fp8):
        return None, None

    device = "cuda"
    torch.manual_seed(42)

    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((N, K), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    # Warmup
    for _ in range(num_warmup):
        matmul_tma_ws_kernel[grid](
            A, B, C,
            *A.stride(), *B.stride(), *C.stride(),
            M, N, K,
            num_stages, BLOCK_M, BLOCK_N, BLOCK_K, 8,
            num_warps=num_warps,
            USE_FP8=use_fp8, A_USE_TMA=a_tma, B_USE_TMA=b_tma)

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        matmul_tma_ws_kernel[grid](
            A, B, C,
            *A.stride(), *B.stride(), *C.stride(),
            M, N, K,
            num_stages, BLOCK_M, BLOCK_N, BLOCK_K, 8,
            num_warps=num_warps,
            USE_FP8=use_fp8, A_USE_TMA=a_tma, B_USE_TMA=b_tma)
        end_events[i].record()

    torch.cuda.synchronize()
    elapsed_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / num_iters

    tflops = 2 * M * N * K / (elapsed_ms * 1e-3) / 1e12
    return tflops, elapsed_ms


def check_correctness(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps=4):
    """Verify correctness against cuBLAS."""
    if exceeds_smem_capacity(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, False):
        return "skip"

    device = "cuda"
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((N, K), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_tma_ws_kernel[grid](
        A, B, C,
        *A.stride(), *B.stride(), *C.stride(),
        M, N, K,
        num_stages, BLOCK_M, BLOCK_N, BLOCK_K, 8,
        num_warps=num_warps,
        USE_FP8=False, A_USE_TMA=True, B_USE_TMA=True)

    C_ref = torch.empty((M, N), dtype=torch.float16, device=device)
    cublas.matmul(A, B, C_ref)

    try:
        torch.testing.assert_close(C_ref, C, atol=0.03, rtol=0.03)
        return "pass"
    except Exception as e:
        return f"fail: {e}"


# Test configurations
# Focus on: small BLOCK_K + large K = many iterations (predication overhead)
# Also: larger shapes for throughput comparison
configs = [
    # Label,          M,    N,    K,   BM,  BN,  BK, stages, warps
    ("k1iter_bk64",   128, 128,   64,  128, 128,  64,  4,  4),   # 1 iteration
    ("k2iter_bk64",   128, 128,  128,  128, 128,  64,  4,  4),   # 2 iterations
    ("k4iter_bk64",   128, 128,  256,  128, 128,  64,  4,  4),   # 4 iterations
    ("k8iter_bk64",   128, 128,  512,  128, 128,  64,  4,  4),   # 8 iterations
    ("k16iter_bk64",  128, 128, 1024,  128, 128,  64,  4,  4),   # 16 iterations
    ("k32iter_bk64",  128, 128, 2048,  128, 128,  64,  4,  4),   # 32 iterations
    ("k64iter_bk64",  128, 128, 4096,  128, 128,  64,  4,  4),   # 64 iterations
    ("k2iter_bk128",  128, 128,  256,  128, 128, 128,  4,  4),   # BK=128, 2 iter
    ("k4iter_bk128",  128, 128,  512,  128, 128, 128,  4,  4),
    ("k8iter_bk128",  128, 128, 1024,  128, 128, 128,  4,  4),
    ("k16iter_bk128", 128, 128, 2048,  128, 128, 128,  4,  4),
    ("k32iter_bk128", 128, 128, 4096,  128, 128, 128,  4,  4),
    ("medium_512",    512, 512,  512,  128, 128,  64,  4,  4),
    ("medium_1k",    1024, 1024, 1024,  128, 128,  64,  4,  4),
    ("large_2k",     2048, 2048,  512,  128, 128,  64,  4,  4),
    ("large_4k",     4096, 4096,  512,  128, 128,  64,  4,  4),
    # Very large shapes - execution time long enough to measure above launch overhead
    ("huge_4k_4k",   4096, 4096, 4096,  128, 128,  64,  4,  4),   # 64 iterations, ~ms range
    ("huge_8k_2k",   8192, 8192, 2048,  128, 128,  64,  4,  4),   # many tiles + many iterations
    ("huge_8k_4k",   8192, 8192, 4096,  128, 128,  64,  4,  8),
    # 3 stages (fewer pipelining stages, different peeling behavior)
    ("k8iter_3stg",   128, 128,  512,  128, 128,  64,  3,  4),
    ("k16iter_3stg",  128, 128, 1024,  128, 128,  64,  3,  4),
]


def run_all(label):
    results = []
    for name, M, N, K, BM, BN, BK, stages, warps in configs:
        print(f"  {name}: M={M} N={N} K={K} BK={BK} stages={stages} ...", flush=True)

        # Check correctness first
        correct = check_correctness(M, N, K, BM, BN, BK, stages, warps)
        if correct == "skip":
            print(f"    {label}: SKIP (smem)", flush=True)
            results.append({"name": name, "M": M, "N": N, "K": K, "BK": BK,
                           "stages": stages, "tflops": 0, "ms": 0,
                           "correct": "skip", "label": label})
            continue

        tflops, ms = benchmark_shape(M, N, K, BM, BN, BK, stages, warps)
        if tflops is None:
            print(f"    {label}: SKIP (smem)", flush=True)
            results.append({"name": name, "M": M, "N": N, "K": K, "BK": BK,
                           "stages": stages, "tflops": 0, "ms": 0,
                           "correct": "skip", "label": label})
            continue

        k_iters = K // BK
        print(f"    {label}: tflops={tflops:.4f} ms={ms:.3f} k_iters={k_iters} correct={correct}", flush=True)
        results.append({"name": name, "M": M, "N": N, "K": K, "BK": BK,
                       "stages": stages, "tflops": tflops, "ms": ms,
                       "k_iters": k_iters, "correct": correct, "label": label})
    return results


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    print(f"Running benchmark with label: {label}", flush=True)
    results = run_all(label)
    outfile = f"bench_epilogue_{label}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {outfile}", flush=True)