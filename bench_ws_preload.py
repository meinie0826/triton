"""Benchmark: WS preload vs WS standard on Blackwell.

Uses the exact same kernel patterns as benchmark_epilogue_peeling.py (which works on B300).
Only tests ws_standard and ws_preload (no persistent for simplicity).
"""
import json
import sys
import os
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip

if not is_hip() and torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    sys.exit("Requires CUDA + Blackwell GPU")

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count


def alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


# ─── Kernel A: Standard WS (exact pattern from benchmark_epilogue_peeling.py) ───
@triton.jit
def matmul_ws_standard(
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
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

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


# ─── Kernel B: WS preload (first load+dot BEFORE WS loop) ───
@triton.jit
def matmul_ws_preload(
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
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    k_tiles = tl.cdiv(K, BLOCK_K)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    # ─── Hoist first iteration: load + dot BEFORE WS loop ───
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a0 = a_desc.load((off_am, 0))
    b0 = b_desc.load((off_bn, 0))
    accumulator = tl.dot(a0, b0.T, accumulator)

    # WS loop from k=1 (first tile already processed)
    if k_tiles > 1:
        for k in tl.range(1, k_tiles, warp_specialize=True, num_stages=num_stages):
            off_k = k * BLOCK_K
            a = a_desc.load((off_am, off_k))
            b = b_desc.load((off_bn, off_k))
            accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    c_desc.store((off_am, off_bn), c)


# ─── Benchmark ───

SHAPES = [
    ("1k",   1024, 1024, 1024),
    ("2k",   2048, 2048, 2048),
    ("4k",   4096, 4096, 4096),
    ("8k4k", 8192, 8192, 4096),
]

TILE_CONFIGS = [
    ("128x128x64_s3_w4",  128, 128, 64,  3, 4),
    ("128x128x64_s4_w4",  128, 128, 64,  4, 4),
    ("128x256x64_s3_w8",  128, 256, 64,  3, 8),
]

cublas_tflops = 0


def run_cublas(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    for _ in range(5):
        cublas.matmul(a, b, c)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    for i in range(50):
        starts[i].record()
        cublas.matmul(a, b, c)
        ends[i].record()
    torch.cuda.synchronize()
    ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / 50
    return ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12


def bench(fn, M, N, K, n_warmup=10, n_repeat=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_repeat
    return ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12


def run_ws_standard(M, N, K, BM, BN, BK, stages, warps):
    triton.set_allocator(alloc_fn)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    def fn():
        matmul_ws_standard[grid](a, b, c,
                                 a.stride(0), a.stride(1),
                                 b.stride(0), b.stride(1),
                                 c.stride(0), c.stride(1),
                                 M, N, K,
                                 num_stages=stages,
                                 BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                 num_warps=warps)
    return bench(fn, M, N, K)


def run_ws_preload(M, N, K, BM, BN, BK, stages, warps):
    triton.set_allocator(alloc_fn)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    def fn():
        matmul_ws_preload[grid](a, b, c,
                                a.stride(0), a.stride(1),
                                b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1),
                                M, N, K,
                                num_stages=stages,
                                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                num_warps=warps)
    return bench(fn, M, N, K)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    if not torch.cuda.is_available():
        sys.exit("No CUDA")

    print(f"GPU: {torch.cuda.get_device_name(0)}, SMs: {NUM_SMS}")

    results = []
    for shape_name, M, N, K in SHAPES:
        global cublas_tflops
        os.system("rm -rf ~/.triton/cache")
        _, cublas_tflops = run_cublas(M, N, K)
        print(f"\n=== {shape_name}: {M}x{N}x{K} === cuBLAS: {cublas_tflops:.3f} TFLOPS")

        for cfg_name, BM, BN, BK, stages, warps in TILE_CONFIGS:
            for variant, run_fn in [("ws_standard", run_ws_standard), ("ws_preload", run_ws_preload)]:
                os.system("rm -rf ~/.triton/cache")
                try:
                    ms, tflops = run_fn(M, N, K, BM, BN, BK, stages, warps)
                    ratio = tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0
                    print(f"  {variant}/{cfg_name}: {tflops:.3f} TFLOPS ({ratio:.1f}% cuBLAS)")
                except Exception as e:
                    tflops = 0; ratio = 0; ms = 0
                    err = str(e)[:100]
                    print(f"  {variant}/{cfg_name}: ERROR ({err})")

                results.append({
                    "label": label, "shape": shape_name, "M": M, "N": N, "K": K,
                    "config": cfg_name, "BM": BM, "BN": BN, "BK": BK,
                    "stages": stages, "warps": warps, "variant": variant,
                    "tflops": tflops, "ms": ms, "ratio_pct": ratio,
                    "cublas_tflops": cublas_tflops,
                })

    outfile = f"bench_ws_preload_{label}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Print summary
    print("\n=== Summary: ws_preload vs ws_standard ===")
    ws_std = {r['shape'] + "|" + r['config']: r for r in results if r['variant'] == 'ws_standard' and r['tflops'] > 0}
    ws_pre = {r['shape'] + "|" + r['config']: r for r in results if r['variant'] == 'ws_preload' and r['tflops'] > 0}
    deltas = []
    for k in sorted(set(ws_std.keys()) & set(ws_pre.keys())):
        s = ws_std[k]
        p = ws_pre[k]
        delta = (p['tflops'] - s['tflops']) / s['tflops'] * 100
        deltas.append(delta)
        print(f"  {k}: std={s['tflops']:.2f} pre={p['tflops']:.2f} Δ={delta:+.2f}% ratio: {s['ratio_pct']:.1f}%→{p['ratio_pct']:.1f}%")
    if deltas:
        print(f"\n  Average delta: {sum(deltas)/len(deltas):+.2f}%")


if __name__ == "__main__":
    main()