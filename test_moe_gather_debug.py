#!/usr/bin/env python3
"""Minimal MoE-style gather test to trace lowering path."""
import torch
import triton
import triton.language as tl

@triton.jit
def moe_gather_kernel(
    Y, YPtr, stride_y_m, stride_y_k,
    X, XPtr, stride_x_z, stride_x_m, stride_x_k,
    W, WPtr, stride_w_e, stride_w_k, stride_w_n,
    GatherIndx,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    X_TMA_MODE: tl.constexpr,
):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Load gather indices
    offs_m = off_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    if X_TMA_MODE == "dense":
        USE_GATHER_TMA: tl.constexpr = True
        offs_x_m = tl.load(GatherIndx + pid_z.to(tl.int64) * M + offs_m, mask=mask_m, other=-1)
        offs_x_m = tl.where(mask_m, offs_x_m, -1)

        # TMA gather
        x = X.gather(offs_x_m, 0)
        x = x.reshape(BLOCK_M, BLOCK_K // 2)

    else:
        USE_GATHER_TMA: tl.constexpr = False
        offs_x_m = tl.load(GatherIndx + pid_z.to(tl.int64) * M + offs_m, mask=mask_m)
        offs_x_k = tl.arange(0, BLOCK_K // 2)[None, :] * stride_x_k
        XPtrs = XPtr + pid_z.to(tl.int64) * stride_x_z + offs_x_m.to(tl.int64)[:, None] * stride_x_m + offs_x_k
        x = tl.load(XPtrs, mask=mask_m[:, None])

    # Simple matmul-like dot (just for completeness, not the focus)
    w = tl.load(WPtr + pid_z.to(tl.int64) * stride_w_e + tl.arange(0, BLOCK_K // 2)[:, None] * stride_w_k + tl.arange(0, BLOCK_N)[None, :] * stride_w_n)
    acc = tl.dot(x, w)

    # Store
    offs_y_m = off_m + tl.arange(0, BLOCK_M)
    offs_y_n = off_n + tl.arange(0, BLOCK_N)
    YPtrs = YPtr + offs_y_m[:, None] * stride_y_m + offs_y_n[None, :] * stride_y_k
    mask = mask_m[:, None] & (offs_y_n < N)[None, :]
    tl.store(YPtrs, acc, mask=mask)


def run_gather_test(X_TMA_MODE=None):
    Z = 1   # num experts
    M = 64  # tokens
    N = 32  # hidden dim
    K = 128

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 128

    X = torch.randn((Z, M, K), dtype=torch.bfloat16, device="cuda")
    W = torch.randn((Z, K, N), dtype=torch.bfloat16, device="cuda")
    Y = torch.empty((M, N), dtype=torch.float32, device="cuda")

    # Gather indices: random permutation of M rows
    GatherIndx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    GatherIndx = GatherIndx[torch.randperm(M, device="cuda")]

    grid = (Z, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    if X_TMA_MODE == "dense":
        X_desc = tl.make_tensor_descriptor(
            X,
            shape=[Z, M, K],
            strides=[M * K, K, 2],
            block_shape=[1, BLOCK_K // 2],
        )
        moe_gather_kernel[grid](
            Y, Y, *Y.stride(),
            X_desc, X, *X.stride(),
            W, W, *W.stride(),
            GatherIndx,
            M, N, K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            X_TMA_MODE="dense",
        )
    else:
        moe_gather_kernel[grid](
            Y, Y, *Y.stride(),
            None, X, *X.stride(),
            W, W, *W.stride(),
            GatherIndx,
            M, N, K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            X_TMA_MODE=None,
        )


if __name__ == "__main__":
    print("=== Testing TMA gather (X_TMA_MODE='dense') ===")
    try:
        run_gather_test(X_TMA_MODE="dense")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

    print("\n=== Testing non-TMA gather ===")
    try:
        run_gather_test(X_TMA_MODE=None)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")