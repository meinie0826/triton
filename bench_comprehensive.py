"""Comprehensive benchmark for Triton matmul kernels on Blackwell.

Tests multiple kernel variants and configurations to understand:
- Whether optimization effects depend on kernel configuration
- How close to cuBLAS each variant gets
- Whether WS first-load overlap would help differently for different configs
"""
import json
import sys
import os
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip, is_blackwell
from triton.tools.tensor_descriptor import TensorDescriptor

if not is_hip() and torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None
    sys.exit("Requires CUDA + Blackwell GPU")

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count


def _compute_pid(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size
    return pid_m, pid_n


def alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


# ─── Kernel 1: Non-WS TMA matmul ───
@triton.jit
def matmul_tma_kernel(
        a_desc, b_desc, c_desc,
        M, N, K,
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

    k_tiles = tl.cdiv(K, BLOCK_K)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(k_tiles, num_stages=3):
        off_k = k * BLOCK_K
        a = a_desc.load([off_am, off_k])
        b = b_desc.load([off_bn, off_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    c_desc.store([off_am, off_bn], c)


# ─── Kernel 2: WS TMA matmul ───
@triton.jit
def matmul_tma_ws_kernel(
        a_desc, b_desc, c_desc,
        M, N, K,
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

    k_tiles = tl.cdiv(K, BLOCK_K)
    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=3):
        off_k = k * BLOCK_K
        a = a_desc.load([off_am, off_k])
        b = b_desc.load([off_bn, off_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    c_desc.store([off_am, off_bn], c)


# ─── Kernel 3: WS TMA persistent matmul ───
@triton.jit
def matmul_tma_ws_persistent_kernel(
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 8,
        NUM_SMS: tl.constexpr,
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
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS)
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

def run_cublas(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    # Warmup
    for _ in range(5):
        cublas.matmul(a, b, c)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(50):
        start.record()
        cublas.matmul(a, b, c)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ms = sum(times) / len(times)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


def benchmark_fn(kernel_fn, n_warmup=10, n_repeat=100):
    for _ in range(n_warmup):
        kernel_fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        start_events[i].record()
        kernel_fn()
        end_events[i].record()
    torch.cuda.synchronize()
    ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / n_repeat
    tflops = 2.0 * M_val * N_val * K_val / (ms * 1e-3) / 1e12
    return ms, tflops


M_val = N_val = K_val = 0  # set by main loop


def run_tma_nonws(M, N, K, BM, BN, BK, stages, warps):
    triton.set_allocator(alloc_fn)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    a_desc = TensorDescriptor.from_outer_outer(a, [BM, BK])
    b_desc = TensorDescriptor.from_outer_outer(b.T, [BN, BK])
    c_desc = TensorDescriptor.from_outer_outer(c, [BM, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    def fn():
        matmul_tma_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                num_stages=stages, num_warps=warps)
    return benchmark_fn(fn)


def run_tma_ws(M, N, K, BM, BN, BK, stages, warps):
    triton.set_allocator(alloc_fn)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    a_desc = TensorDescriptor.from_outer_outer(a, [BM, BK])
    b_desc = TensorDescriptor.from_outer_outer(b.T, [BN, BK])
    c_desc = TensorDescriptor.from_outer_outer(c, [BM, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    def fn():
        matmul_tma_ws_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                                   BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                   num_stages=stages, num_warps=warps)
    return benchmark_fn(fn)


def run_tma_ws_persistent(M, N, K, BM, BN, BK, stages, warps, subtile=False):
    triton.set_allocator(alloc_fn)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    a_desc = TensorDescriptor.from_outer_outer(a, [BM, BK])
    b_desc = TensorDescriptor.from_outer_outer(b.T, [BN, BK])
    BN_store = BN // 2 if subtile else BN
    c_desc = TensorDescriptor.from_outer_outer(c, [BM, BN_store])
    grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)

    def fn():
        matmul_tma_ws_persistent_kernel[grid](
            a_desc, b_desc, c_desc, M, N, K,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            NUM_SMS=NUM_SMS, EPILOGUE_SUBTILE=subtile,
            num_stages=stages, num_warps=warps)
    return benchmark_fn(fn)


def dump_ir(M, N, K, BM, BN, BK, stages, warps, kernel_name="ws"):
    """Extract IR info from kernel compilation."""
    triton.set_allocator(alloc_fn)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    a_desc = TensorDescriptor.from_outer_outer(a, [BM, BK])
    b_desc = TensorDescriptor.from_outer_outer(b.T, [BN, BK])
    c_desc = TensorDescriptor.from_outer_outer(c, [BM, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    if kernel_name == "ws":
        matmul_tma_ws_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                                   BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                   num_stages=stages, num_warps=warps)
    else:
        matmul_tma_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                num_stages=stages, num_warps=warps)

    info = {}
    asm_data = c.device_astributes if hasattr(c, 'device_attributes') else {}
    # Try to get IR from kernel cache
    import re
    try:
        kernel = matmul_tma_ws_kernel if kernel_name == "ws" else matmul_tma_kernel
        ttgir = kernel.asm["ttgir"]
        info["has_warp_specialize"] = "warp_specialize" in ttgir
        m = re.findall(r'requestedRegisters\s*=\s*array<i32:\s*([^>]+)>', ttgir)
        if m:
            info["requested_registers"] = m[0].strip()
        m = re.findall(r'num_warps\((\d+)\)', ttgir)
        if m:
            info["partition_num_warps"] = m
        m = re.findall(r'tc_gen5_mma', ttgir)
        info["tc_gen5_mma_count"] = len(m)
        m = re.findall(r'setmaxnreg', ttgir)
        info["setmaxnreg_count"] = len(m)
    except Exception as e:
        info["ir_error"] = str(e)
    return info


# ─── Test configurations ───

# Shape configs
SHAPES = [
    ("medium_1k",   1024, 1024, 1024),
    ("large_2k",    2048, 2048, 2048),
    ("large_4k",    4096, 4096, 4096),
    ("huge_8k",     8192, 8192, 4096),
]

# Tile configs (BM, BN, BK, stages, warps)
TILE_CONFIGS = [
    ("small_128x128x64_s3_w4",   128, 128, 64,  3, 4),
    ("small_128x128x64_s4_w4",   128, 128, 64,  4, 4),
    ("medium_128x256x64_s3_w8",  128, 256, 64,  3, 8),
    ("medium_128x256x64_s4_w8",  128, 256, 64,  4, 8),
    ("large_128x128x128_s3_w4",  128, 128, 128, 3, 4),
]

# Kernel variants
VARIANTS = [
    ("tma_nonws",       run_tma_nonws),
    ("tma_ws",          run_tma_ws),
    ("tma_ws_persist",  run_tma_ws_persistent),
    ("tma_ws_persist_subtile", lambda M,N,K,BM,BN,BK,s,w: run_tma_ws_persistent(M,N,K,BM,BN,BK,s,w,subtile=True)),
]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    dump_ir_flag = "--dump-ir" in sys.argv

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device"); sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}, SMs: {NUM_SMS}")

    results = []
    for shape_name, M, N, K in SHAPES:
        print(f"\n=== {shape_name}: {M}x{N}x{K} ===")

        # cuBLAS baseline
        os.system("rm -rf ~/.triton/cache")
        cublas_ms, cublas_tflops = run_cublas(M, N, K)
        print(f"  cuBLAS: {cublas_ms:.3f} ms, {cublas_tflops:.3f} TFLOPS")

        for cfg_name, BM, BN, BK, stages, warps in TILE_CONFIGS:
            # Check SMEM capacity
            smem_per_stage = BK * (BM + BN) * 2  # FP16
            smem_total = stages * smem_per_stage + BM * BN * 4  # accumulator in SMEM for non-TMA
            if smem_total > 228 * 1024:  # B200 has ~228KB SMEM
                print(f"  {cfg_name}: SKIP (SMEM overflow)")
                continue

            for var_name, var_fn in VARIANTS:
                os.system("rm -rf ~/.triton/cache")
                global M_val, N_val, K_val
                M_val, N_val, K_val = M, N, K

                try:
                    ms, tflops = var_fn(M, N, K, BM, BN, BK, stages, warps)
                    ratio = tflops / cublas_tflops * 100
                    print(f"  {var_name}/{cfg_name}: {ms:.3f} ms, {tflops:.3f} TFLOPS ({ratio:.1f}% cuBLAS)")
                except Exception as e:
                    tflops = 0
                    ms = 0
                    ratio = 0
                    print(f"  {var_name}/{cfg_name}: FAIL ({e})")

                result = {
                    "label": label,
                    "shape": shape_name, "M": M, "N": N, "K": K,
                    "config": cfg_name, "BM": BM, "BN": BN, "BK": BK,
                    "stages": stages, "warps": warps,
                    "variant": var_name,
                    "tflops": tflops, "ms": ms,
                    "cublas_tflops": cublas_tflops,
                    "ratio_pct": ratio,
                }

                if dump_ir_flag and var_name in ("tma_ws", "tma_nonws"):
                    os.system("rm -rf ~/.triton/cache")
                    ir_info = dump_ir(M, N, K, BM, BN, BK, stages, warps,
                                      "ws" if var_name == "tma_ws" else "nonws")
                    result.update(ir_info)
                    for key in ("has_warp_specialize", "requested_registers",
                                "partition_num_warps", "tc_gen5_mma_count", "setmaxnreg_count"):
                        if key in ir_info:
                            print(f"    {key}: {ir_info[key]}")

                results.append(result)

    outfile = f"bench_comprehensive_{label}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()