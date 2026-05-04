"""Minimal WGMMA diagnostic for Blackwell GPU.

All tensors use explicit layouts to bypass auto_encoding resolution
(which fails on sm_100). This isolates the wgmma-specific crash.

Step 1: Basic Gluon with explicit BlockedLayout — does it compile on sm_100?
Step 2: NVMMADistributedLayout declaration only — does layout trigger crash?
Step 3: Full warpgroup_mma call — does wgmma PTX generation crash?
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma,
)

M, N, K = 64, 32, 32
SIZE = 1024

# Common explicit layout for 1D vectors (mimics working scalar megakernel)
BLOCKED_1D: gl.constexpr = gl.BlockedLayout([SIZE // 128], [32], [4], [0])


# ---------------------------------------------------------------------------
# Step 1: Basic Gluon with explicit BlockedLayout (no wgmma)
# ---------------------------------------------------------------------------

@gluon.jit
def step1_basic(a_ptr, d_ptr, SIZE: gl.constexpr):
    idx = gl.arange(0, SIZE, layout=BLOCKED_1D)
    a = gl.load(a_ptr + idx.to(gl.int64))
    gl.store(d_ptr + idx.to(gl.int64), a * 2.0)


def test_step1():
    A = torch.randn(SIZE, device="cuda", dtype=torch.float16)
    D = torch.empty_like(A)
    step1_basic[(1,)](A, D, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(D, A * 2.0, rtol=1e-2, atol=1e-2)
    print("Step 1 PASSED: Basic Gluon (explicit BlockedLayout) works on sm_100")


# ---------------------------------------------------------------------------
# Step 2: NVMMADistributedLayout only (declare accumulator, no warpgroup_mma)
# ---------------------------------------------------------------------------

@gluon.jit
def step2_layout_only(c_ptr, d_ptr, M: gl.constexpr, N: gl.constexpr):
    idx = gl.arange(0, M * N, layout=BLOCKED_1D)
    # Declare NVMMADistributedLayout but DON'T call warpgroup_mma
    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 32, 16],
    )
    acc = gl.zeros((M, N), dtype=gl.float32, layout=acc_layout)
    # Just pass through input (acc is unused)
    c = gl.load(c_ptr + idx.to(gl.int64))
    gl.store(d_ptr + idx.to(gl.int64), c)


def test_step2():
    C = torch.randn(M * N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    step2_layout_only[(1,)](C, D, M=M, N=N, num_warps=4)
    torch.testing.assert_close(D, C, rtol=1e-2, atol=1e-2)
    print("Step 2 PASSED: NVMMADistributedLayout declaration works on sm_100")


# ---------------------------------------------------------------------------
# Step 3: Full warpgroup_mma
# ---------------------------------------------------------------------------

@gluon.jit
def step3_wgmma(a_ptr, b_ptr, c_ptr, d_ptr,
                M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    # Load with explicit layouts
    a = gl.load(a_ptr + gl.arange(0, M * K, layout=BLOCKED_1D).to(gl.int64))
    b = gl.load(b_ptr + gl.arange(0, K * N, layout=BLOCKED_1D).to(gl.int64))
    c = gl.load(c_ptr + gl.arange(0, M * N, layout=BLOCKED_1D).to(gl.int64))

    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 32, 16],
    )
    acc = gl.zeros((M, N), dtype=gl.float32, layout=acc_layout)
    # This is the line that may crash on Blackwell
    acc = warpgroup_mma(a, b, acc)
    gl.store(d_ptr + gl.arange(0, M * N, layout=BLOCKED_1D).to(gl.int64), acc)


def test_step3():
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M * N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    step3_wgmma[(1,)](A, B, C, D, M=M, N=N, K=K, num_warps=4)
    expected = (A @ B).flatten() + C
    torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)
    print("Step 3 PASSED: warpgroup_mma works on sm_100!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    print(f"GPU capability: sm_{cap[0]}{cap[1]}")

    print("\n=== Step 1: Basic Gluon (explicit layout) ===")
    test_step1()

    print("\n=== Step 2: NVMMADistributedLayout only ===")
    test_step2()

    print("\n=== Step 3: Full warpgroup_mma ===")
    test_step3()

    print("\nAll steps passed!")