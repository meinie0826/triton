#!/usr/bin/env python3
"""Minimal MoE-style gather test to trace lowering path."""
import torch
import triton
import triton.language as tl


# ---- Test 1: tl.gather (non-TMA, triggers GatherOp lowering) ----
@triton.jit
def gather_kernel(
    out_ptr, stride_out_m,
    src_ptr, stride_src_z, stride_src_m, stride_src_k,
    gather_indx_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    Z: tl.constexpr,
):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)

    off_m = pid_m * BLOCK_M

    # Load source: [BLOCK_M, BLOCK_N]
    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    src_ptrs = src_ptr + pid_z.to(tl.int64) * stride_src_z + offs_m[:, None] * stride_src_m + offs_n[None, :] * stride_src_k
    src = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Load gather indices
    gather_indx = tl.load(gather_indx_ptr + pid_z.to(tl.int64) * M + offs_m, mask=mask_m, other=0)

    # Perform gather: src[gather_indx, :]
    out = tl.gather(src, gather_indx, axis=0)

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :]
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def test_tl_gather():
    Z = 1
    M = 64
    N = 32
    BLOCK_M = 64
    BLOCK_N = 32

    src = torch.randn((Z, M, N), dtype=torch.float32, device="cuda")
    gather_indx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    gather_indx = gather_indx[torch.randperm(M, device="cuda")]
    out = torch.empty((M, N), dtype=torch.float32, device="cuda")

    grid = (Z, triton.cdiv(M, BLOCK_M))
    gather_kernel[grid](
        out, *out.stride(),
        src, *src.stride(),
        gather_indx,
        M, N, BLOCK_M, BLOCK_N, Z,
    )
    torch.cuda.synchronize()

    # Verify
    ref = src[0, gather_indx, :]
    torch.testing.assert_close(out, ref)
    print("tl.gather test PASSED")


# ---- Test 2: TMA descriptor gather (DescriptorGatherOp lowering) ----
@triton.jit
def tma_gather_kernel(
    out_ptr, stride_out_m,
    X_desc,
    gather_indx_ptr,
    M: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Z: tl.constexpr,
):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)

    off_m = pid_m * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Load gather indices
    gather_indx = tl.load(gather_indx_ptr + pid_z.to(tl.int64) * M + offs_m, mask=mask_m, other=-1)
    gather_indx = tl.where(mask_m, gather_indx, -1)

    # TMA gather
    x = X_desc.gather(gather_indx, 0)
    x = x.reshape(BLOCK_M, BLOCK_K)

    # Store result (just the first N columns for simplicity)
    N = min(BLOCK_K, 32)
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + tl.arange(0, N)[None, :]
    tl.store(out_ptrs, x[:, :N], mask=mask_m[:, None])


def test_tma_gather():
    Z = 1
    M = 64
    K = 128
    BLOCK_M = 64
    BLOCK_K = 128

    X = torch.randn((Z, M, K), dtype=torch.bfloat16, device="cuda")
    gather_indx = torch.arange(0, M, dtype=torch.int32, device="cuda")
    gather_indx = gather_indx[torch.randperm(M, device="cuda")]
    out = torch.empty((M, 32), dtype=torch.float32, device="cuda")

    X_desc = tl.make_tensor_descriptor(
        X,
        shape=[Z, M, K],
        strides=[M * K, K, 1],
        block_shape=[1, BLOCK_K],
    )

    grid = (Z, triton.cdiv(M, BLOCK_M))
    try:
        tma_gather_kernel[grid](
            out, *out.stride(),
            X_desc,
            gather_indx,
            M, BLOCK_M, BLOCK_K, Z,
        )
        torch.cuda.synchronize()
        print("TMA gather test PASSED")
    except Exception as e:
        print(f"TMA gather test FAILED: {e}")


if __name__ == "__main__":
    print("=== Test 1: tl.gather (GatherOp lowering) ===")
    test_tl_gather()

    print("\n=== Test 2: TMA descriptor gather (DescriptorGatherOp lowering) ===")
    test_tma_gather()