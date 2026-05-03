"""Benchmark: WS first-load hoisting vs standard WS.

Kernel variant 'ws_preload' does the first TMA load + dot BEFORE the WS loop,
so MMA partition doesn't need to wait_barrier on the first iteration.

Kernel variant 'ws_standard' is the standard WS kernel (your original version).

Both share the same tile/stage configs for fair comparison.
"""
import json
import sys
import os
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip
from triton.tools.tensor_descriptor import TensorDescriptor

if not is_hip() and torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    sys.exit("Requires CUDA + Blackwell GPU")

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count


def _compute_pid(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size
    return pid_m, pid_n


def alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


# ─── Kernel A: Standard WS (baseline) ───
@triton.jit
def matmul_ws_standard(
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    pid_m, pid_n = _compute_pid(pid, num_pid_in_group, num_pid_m, GROUP_M)

    k_tiles = tl.cdiv(K, BLOCK_K)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Standard WS: first TMA load happens inside the loop, MMA must wait
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=3):
        off_k = k * BLOCK_K
        a = a_desc.load([off_am, off_k])
        b = b_desc.load([off_bn, off_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    c_desc.store([off_am, off_bn], c)


# ─── Kernel B: WS with first load hoisted (preload) ───
@triton.jit
def matmul_ws_preload(
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    pid_m, pid_n = _compute_pid(pid, num_pid_in_group, num_pid_m, GROUP_M)

    k_tiles = tl.cdiv(K, BLOCK_K)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    # ─── Hoist first iteration: load + dot BEFORE WS loop ───
    # This runs in the default warp group, before partitioning.
    # MMA partition enters the loop with accumulator already initialized.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a0 = a_desc.load([off_am, 0])
    b0 = b_desc.load([off_bn, 0])
    accumulator = tl.dot(a0, b0.T, accumulator)

    # WS loop starts from k=1 (first tile already processed)
    if k_tiles > 1:
        for k in tl.range(1, k_tiles, warp_specialize=True, num_stages=3):
            off_k = k * BLOCK_K
            a = a_desc.load([off_am, off_k])
            b = b_desc.load([off_bn, off_k])
            accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    c_desc.store([off_am, off_bn], c)


# ─── Kernel C: WS persistent + epilogue subtiling ───
@triton.jit
def matmul_ws_persistent(
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 8,
        NUM_SMS: tl.constexpr = 128,
        EPILOGUE_SUBTILE: tl.constexpr = False,
):
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M)
        offs_am_c = pid_m * BLOCK_M
        offs_bn_c = pid_n * BLOCK_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c)


# ─── Kernel D: WS persistent + preload ───
@triton.jit
def matmul_ws_persistent_preload(
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 8,
        NUM_SMS: tl.constexpr = 128,
        EPILOGUE_SUBTILE: tl.constexpr = False,
):
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        # ─── Hoist first iteration ───
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a0 = a_desc.load([offs_am, 0])
        b0 = a_desc.load([offs_bn, 0])
        accumulator = tl.dot(a0, b0.T, accumulator)

        if k_tiles > 1:
            for ki in range(1, k_tiles):
                offs_k = ki * BLOCK_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M)
        offs_am_c = pid_m * BLOCK_M
        offs_bn_c = pid_n * BLOCK_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c)


# ─── Benchmark infrastructure ───

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
    ("128x256x64_s4_w8",  128, 256, 64,  4, 8),
]

M_val = N_val = K_val = 0


def run_cublas(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
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


def make_descs(M, N, K, BM, BN, BK, BN_store=None):
    if BN_store is None:
        BN_store = BN
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    a_desc = TensorDescriptor.from_outer_outer(a, [BM, BK])
    b_desc = TensorDescriptor.from_outer_outer(b.T, [BN, BK])
    c_desc = TensorDescriptor.from_outer_outer(c, [BM, BN_store])
    return a, b, c, a_desc, b_desc, c_desc


def run_variant(variant, M, N, K, BM, BN, BK, stages, warps):
    triton.set_allocator(alloc_fn)
    os.system("rm -rf ~/.triton/cache")

    if variant == "ws_standard":
        a, b, c, ad, bd, cd = make_descs(M, N, K, BM, BN, BK)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        def fn():
            matmul_ws_standard[grid](ad, bd, cd, M, N, K,
                                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                     num_stages=stages, num_warps=warps)
        ms, tf = bench(fn, M, N, K)

    elif variant == "ws_preload":
        a, b, c, ad, bd, cd = make_descs(M, N, K, BM, BN, BK)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        def fn():
            matmul_ws_preload[grid](ad, bd, cd, M, N, K,
                                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                    num_stages=stages, num_warps=warps)
        ms, tf = bench(fn, M, N, K)

    elif variant == "ws_persistent":
        a, b, c, ad, bd, cd = make_descs(M, N, K, BM, BN, BK)
        grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        def fn():
            matmul_ws_persistent[grid](ad, bd, cd, M, N, K,
                                       BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                       NUM_SMS=NUM_SMS,
                                       num_stages=stages, num_warps=warps)
        ms, tf = bench(fn, M, N, K)

    elif variant == "ws_persistent_subtile":
        BN_store = BN // 2
        a, b, c, ad, bd, cd = make_descs(M, N, K, BM, BN, BK, BN_store)
        grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        def fn():
            matmul_ws_persistent[grid](ad, bd, cd, M, N, K,
                                       BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                       NUM_SMS=NUM_SMS, EPILOGUE_SUBTILE=True,
                                       num_stages=stages, num_warps=warps)
        ms, tf = bench(fn, M, N, K)

    elif variant == "ws_persistent_preload":
        a, b, c, ad, bd, cd = make_descs(M, N, K, BM, BN, BK)
        grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        def fn():
            matmul_ws_persistent_preload[grid](ad, bd, cd, M, N, K,
                                               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                               NUM_SMS=NUM_SMS,
                                               num_stages=stages, num_warps=warps)
        ms, tf = bench(fn, M, N, K)

    elif variant == "ws_persistent_preload_subtile":
        BN_store = BN // 2
        a, b, c, ad, bd, cd = make_descs(M, N, K, BM, BN, BK, BN_store)
        grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
        def fn():
            matmul_ws_persistent_preload[grid](ad, bd, cd, M, N, K,
                                               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                               NUM_SMS=NUM_SMS, EPILOGUE_SUBTILE=True,
                                               num_stages=stages, num_warps=warps)
        ms, tf = bench(fn, M, N, K)

    else:
        return 0, 0, 0

    return ms, tf, tf / cublas_tflops * 100 if cublas_tflops > 0 else 0


# Global for ratio calculation
cublas_tflops = 0

VARIANTS = [
    "ws_standard",
    "ws_preload",
    "ws_persistent",
    "ws_persistent_subtile",
    "ws_persistent_preload",
    "ws_persistent_preload_subtile",
]


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
            for variant in VARIANTS:
                try:
                    ms, tflops, ratio = run_variant(variant, M, N, K, BM, BN, BK, stages, warps)
                    if tflops > 0:
                        print(f"  {variant}/{cfg_name}: {tflops:.3f} TFLOPS ({ratio:.1f}% cuBLAS)")
                    else:
                        print(f"  {variant}/{cfg_name}: SKIP/FAIL")
                except Exception as e:
                    tflops = 0; ratio = 0; ms = 0
                    print(f"  {variant}/{cfg_name}: ERROR ({e})")

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


if __name__ == "__main__":
    main()