#!/usr/bin/env python3
"""
Reproduce the _p_matmul.py TMA gather path to see why it doesn't lower to gather4.
We replicate the exact same pattern:
  - X is a TensorDescriptor passed into kernel
  - offs_x_m is loaded from GatherIndx (1D [BLOCK_M], standard blocked layout)
  - x = X.gather(offs_x_m, y_offset)  --> DescriptorGatherOp
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Replicate exactly what _p_matmul.py does in USE_GATHER_TMA branch
@triton.jit
def p_matmul_gather_repro(
    X,                    # TensorDescriptor passed in
    GatherIndx,           # int32 pointer to gather indices
    out_ptr,              # output pointer
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M

    offs_m = off_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Exactly as in _p_matmul.py line 276-279
    offs_x_m = tl.load(GatherIndx + offs_m, mask=mask_m)
    offs_x_m = tl.where(mask_m, offs_x_m, -1)

    # Exactly as in _p_matmul.py line 323
    x = X.gather(offs_x_m, 0)   # DescriptorGatherOp

    # Store result to verify correctness
    tl.store(out_ptr + offs_m[:, None] * K + tl.arange(0, BLOCK_K)[None, :],
             x, mask=mask_m[:, None])


def run():
    M, K = 128, 128
    BLOCK_M, BLOCK_K = 64, 128

    X_data = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    gather_indx = torch.randperm(M, dtype=torch.int32, device="cuda")
    out = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")

    # TensorDescriptor: block_shape[0] must be 1 for gather (per semantic.py:1143)
    # In _p_matmul.py, swizzle_block_shape() handles this automatically
    X_desc = TensorDescriptor(
        X_data,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[1, BLOCK_K],
    )

    grid = (triton.cdiv(M, BLOCK_M),)
    p_matmul_gather_repro[grid](
        X_desc, gather_indx, out,
        M=M, K=K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()

    # Verify
    expected = X_data[gather_indx, :]
    torch.testing.assert_close(out, expected)
    print("Gather repro PASSED")


if __name__ == "__main__":
    print("=== Reproducing _p_matmul.py TMA gather path ===")
    run()