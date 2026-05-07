#!/usr/bin/env python3
"""Minimal gather test to trace lowering path."""
import torch
import triton
import triton.language as tl


# ---- Test 1: tl.gather 1D ----
@triton.jit
def gather_1d_kernel(out_ptr, src_ptr, idx_ptr,
                     AXIS: tl.constexpr, N: tl.constexpr, M: tl.constexpr):
    src = tl.load(src_ptr + tl.arange(0, N))
    idx = tl.load(idx_ptr + tl.arange(0, M))
    out = tl.gather(src, idx, AXIS)
    tl.store(out_ptr + tl.arange(0, M), out)


def test_tl_gather_1d():
    N, M = 32, 64
    src = torch.randn(N, dtype=torch.float32, device="cuda")
    idx = torch.randint(0, N, (M,), dtype=torch.int32, device="cuda")
    out = torch.empty(M, dtype=torch.float32, device="cuda")
    gather_1d_kernel[(1,)](out, src, idx, AXIS=0, N=N, M=M)
    torch.cuda.synchronize()
    expected = torch.gather(src, 0, idx)
    torch.testing.assert_close(out, expected)
    print("tl.gather 1D test PASSED")


# ---- Test 2: TMA descriptor gather (DescriptorGatherOp → AsyncTMAGatherOp) ----
@triton.jit
def tma_gather_kernel(out_ptr, X_desc, gather_indx_ptr,
                      M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # This is the same pattern as _p_matmul.py:
    # offs_x_m is 1D [BLOCK_M], standard blocked layout (NOT warp-broadcast)
    offs_x_m = tl.load(gather_indx_ptr + offs_m, mask=mask_m, other=-1)
    offs_x_m = tl.where(mask_m, offs_x_m, -1)

    # DescriptorGatherOp: x_offsets must satisfy gather4 constraints
    x = X_desc.gather(offs_x_m, 0)  # -> AsyncTMAGatherOp -> cp.async.bulk.tensor.2d.tile::gather4

    # Store first 32 cols
    tl.store(out_ptr + offs_m[:, None] * 32 + tl.arange(0, 32)[None, :],
             x[:, :32], mask=mask_m[:, None])


def test_tma_gather():
    M, K = 64, 128
    BLOCK_M, BLOCK_K = 64, 128

    X = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    gather_indx = torch.randperm(M, dtype=torch.int32, device="cuda")
    out = torch.empty((M, 32), dtype=torch.float32, device="cuda")

    # make_tensor_descriptor must be called OUTSIDE the kernel
    X_desc = tl.make_tensor_descriptor(
        X,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    grid = (triton.cdiv(M, BLOCK_M),)
    try:
        tma_gather_kernel[grid](out, X_desc, gather_indx, M, BLOCK_M, BLOCK_K)
        torch.cuda.synchronize()
        print("TMA gather test PASSED")
    except Exception as e:
        print(f"TMA gather test FAILED: {e}")


if __name__ == "__main__":
    print("=== Test 1: tl.gather 1D ===")
    test_tl_gather_1d()

    print("\n=== Test 2: TMA descriptor gather ===")
    test_tma_gather()