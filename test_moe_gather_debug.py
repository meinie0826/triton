#!/usr/bin/env python3
"""Minimal gather test to trace lowering path."""
import torch
import triton
import triton.language as tl


# ---- Test 1: tl.gather 1D (triggers GatherOp lowering) ----
@triton.jit
def gather_1d_kernel(
    out_ptr,
    src_ptr,
    idx_ptr,
    AXIS: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    src = tl.load(src_ptr + tl.arange(0, N))
    idx = tl.load(idx_ptr + tl.arange(0, M))
    out = tl.gather(src, idx, AXIS)
    tl.store(out_ptr + tl.arange(0, M), out)


def test_tl_gather_1d():
    N = 32
    M = 64
    src = torch.randn(N, dtype=torch.float32, device="cuda")
    idx = torch.randint(0, N, (M,), dtype=torch.int32, device="cuda")
    out = torch.empty(M, dtype=torch.float32, device="cuda")

    gather_1d_kernel[(1,)](out, src, idx, AXIS=0, N=N, M=M)
    torch.cuda.synchronize()

    expected = torch.gather(src, 0, idx)
    torch.testing.assert_close(out, expected)
    print("tl.gather 1D test PASSED")


# ---- Test 2: tl.gather 2D ----
@triton.jit
def gather_2d_kernel(
    out_ptr, out_stride0, out_stride1,
    src_ptr, src_stride0, src_stride1,
    idx_ptr, idx_stride0, idx_stride1,
    AXIS: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    IM: tl.constexpr, IN: tl.constexpr,
):
    src_offs = tl.arange(0, M)[:, None] * src_stride0 + tl.arange(0, N)[None, :] * src_stride1
    src = tl.load(src_ptr + src_offs)

    idx_offs = tl.arange(0, IM)[:, None] * idx_stride0 + tl.arange(0, IN)[None, :] * idx_stride1
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, AXIS)

    out_offs2 = tl.arange(0, IM)[:, None] * out_stride0 + tl.arange(0, IN)[None, :] * out_stride1
    tl.store(out_ptr + out_offs2, out)


def test_tl_gather_2d():
    M, N = 32, 32
    IM, IN = 64, 32
    src = torch.randn(M, N, dtype=torch.float32, device="cuda")
    idx = torch.randint(0, M, (IM, IN), dtype=torch.int32, device="cuda")
    out = torch.empty(IM, IN, dtype=torch.float32, device="cuda")

    gather_2d_kernel[(1,)](out, *out.stride(), src, *src.stride(), idx, *idx.stride(),
                           AXIS=0, M=M, N=N, IM=IM, IN=IN)
    torch.cuda.synchronize()

    expected = torch.gather(src, 0, idx)
    torch.testing.assert_close(out, expected)
    print("tl.gather 2D test PASSED")


# ---- Test 3: TMA descriptor gather ----
@triton.jit
def tma_gather_kernel(
    out_ptr,
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

    # Store result
    offs_n = tl.arange(0, min(BLOCK_K, 32))
    tl.store(out_ptr + offs_m[:, None] * 32 + offs_n[None, :], x[:, :32], mask=mask_m[:, None])


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
            out, X_desc, gather_indx,
            M, BLOCK_M, BLOCK_K, Z,
        )
        torch.cuda.synchronize()
        print("TMA gather test PASSED")
    except Exception as e:
        print(f"TMA gather test FAILED: {e}")


if __name__ == "__main__":
    print("=== Test 1: tl.gather 1D ===")
    test_tl_gather_1d()

    print("\n=== Test 2: tl.gather 2D ===")
    test_tl_gather_2d()

    print("\n=== Test 3: TMA descriptor gather ===")
    test_tma_gather()