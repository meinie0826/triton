"""
Test to verify TCGen5MMAScaledOp data-value classification in warp specialization.

This test:
1. Compiles a dot_scaled kernel with warp_specialize=True
2. Dumps IR and checks tc_gen5_mma_scaled inside ttg.warp_specialize
3. Verifies correctness
4. Benchmarks warp_specialize vs non-warp_specialize performance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

# Reuse data setup from the original tutorial
from importlib import import_module
tutorial = import_module("10-block-scaled-matmul")
initialize_block_scaled = tutorial.initialize_block_scaled
block_scaled_matmul_kernel = tutorial.block_scaled_matmul_kernel


def is_cuda_blackwell():
    return triton.runtime.driver.active.get_current_target().backend == "cuda" and \
           torch.cuda.get_device_capability()[0] in [10, 11]


BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 256
VEC_SIZE = 16  # nvfp4


@triton.jit
def scaled_matmul_ws_kernel(
    a_desc, a_scale_desc,
    b_desc, b_scale_desc,
    c_desc,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    output_type: tl.constexpr,
    ELEM_PER_BYTE_A: tl.constexpr, ELEM_PER_BYTE_B: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr, rep_n: tl.constexpr, rep_k: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_WARP_SPECIALIZE: tl.constexpr,
):
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K),
                      num_stages=NUM_STAGES,
                      warp_specialize=USE_WARP_SPECIALIZE):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def compile_and_check(M, N, K, use_ws, dump_ir=False):
    """Compile kernel, check correctness, optionally dump IR."""
    results = initialize_block_scaled(M, N, K, "nvfp4", compute_reference=True)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference = results[:9]

    output = torch.empty((M, N), dtype=torch.float16, device="cuda")
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])
    dtype_dst = 1  # fp16

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    h = scaled_matmul_ws_kernel[grid](
        a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
        M, N, K, dtype_dst,
        configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
        rep_m, rep_n, rep_k, configs["num_stages"],
        USE_WARP_SPECIALIZE=use_ws,
    )

    # Correctness check
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    label = "warp_specialize" if use_ws else "no_warp_specialize"
    print(f"✅ Correctness passed ({label})")

    if dump_ir:
        ttgir = h.asm["ttgir"]
        print(f"\n{'='*60}")
        print(f"TTGIR key lines ({label}):")
        print(f"{'='*60}")
        ws_found = False
        mma_scaled_found = False
        for line in ttgir.split("\n"):
            if "warp_specialize" in line:
                ws_found = True
                print(line)
            if "tc_gen5_mma_scaled" in line:
                mma_scaled_found = True
                print(line)
            if "requestedRegisters" in line:
                print(line)
        print(f"\nSummary: ttg.warp_specialize={ws_found}, ttng.tc_gen5_mma_scaled={mma_scaled_found}")

    return h


def benchmark(M, N, K, use_ws, reps=100, warmup=20):
    """Simple timing benchmark."""
    results = initialize_block_scaled(M, N, K, "nvfp4", compute_reference=False)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, _ = results[:9]

    output = torch.empty((M, N), dtype=torch.float16, device="cuda")
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])
    dtype_dst = 1

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    # Warmup
    for _ in range(warmup):
        scaled_matmul_ws_kernel[grid](
            a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
            M, N, K, dtype_dst,
            configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
            configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
            rep_m, rep_n, rep_k, configs["num_stages"],
            USE_WARP_SPECIALIZE=use_ws,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        scaled_matmul_ws_kernel[grid](
            a_desc, a_scale_desc, b_desc, b_scale_desc, c_desc,
            M, N, K, dtype_dst,
            configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
            configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
            rep_m, rep_n, rep_k, configs["num_stages"],
            USE_WARP_SPECIALIZE=use_ws,
        )
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / reps
    tflops = 2.0 * M * N * K / ms / 1e12
    label = "warp_specialize" if use_ws else "no_warp_specialize"
    print(f"{label}: {ms:.3f} ms, {tflops:.3f} TFLOPS")
    return ms, tflops


if __name__ == "__main__":
    if not is_cuda_blackwell():
        print("⛔ Requires Blackwell GPU (sm_100/sm_110)")
    else:
        M, N, K = 4096, 4096, 4096

        # 1. IR dump & correctness for warp_specialize=True
        print("="*60)
        print("Phase 1: IR dump + correctness (warp_specialize=True)")
        print("="*60)
        compile_and_check(M, N, K, use_ws=True, dump_ir=True)

        # 2. IR dump & correctness for warp_specialize=False
        print("\n" + "="*60)
        print("Phase 2: IR dump + correctness (warp_specialize=False)")
        print("="*60)
        compile_and_check(M, N, K, use_ws=False, dump_ir=True)

        # 3. Performance comparison
        print("\n" + "="*60)
        print("Phase 3: Performance benchmark")
        print("="*60)
        ms_ws, tflops_ws = benchmark(M, N, K, use_ws=True)
        ms_no_ws, tflops_no_ws = benchmark(M, N, K, use_ws=False)
        ratio = ms_ws / ms_no_ws
        print(f"\nWS/No-WS time ratio: {ratio:.3f} (1.0 = equal, <1.0 = WS faster)")