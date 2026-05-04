"""Minimal WGMMA diagnostic for Blackwell GPU.

Follows the official Triton Gluon wgmma tutorial pattern:
- TMA load A/B into shared memory (NVMMASharedLayout)
- Load C into registers (NVMMADistributedLayout)
- Issue warpgroup_mma
- Wait and store result

Step 1: Basic Gluon with BlockedLayout → should PASS
Step 2: NVMMADistributedLayout declaration → should PASS  
Step 3: Full WGMMA with TMA + NVMMASharedLayout → tests the real WGMMA path
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


M, N, K = 64, 128, 128
INSTR_SHAPE_N: gl.constexpr = 64


# ---------------------------------------------------------------------------
# Step 1: Basic Gluon (explicit BlockedLayout) — sanity check
# ---------------------------------------------------------------------------

@gluon.jit
def step1_basic(ptr, SIZE: gl.constexpr):
    BLOCKED: gl.constexpr = gl.BlockedLayout([SIZE // 128], [32], [4], [0])
    a = gl.load(ptr + gl.arange(0, SIZE, layout=BLOCKED).to(gl.int64))
    gl.store(ptr + gl.arange(0, SIZE, layout=BLOCKED).to(gl.int64), a * 2.0)


def test_step1():
    A = torch.randn(1024, device="cuda", dtype=torch.float16)
    step1_basic[(1,)](A, SIZE=1024, num_warps=4)
    print("Step 1 PASSED: Basic Gluon works on this GPU")


# ---------------------------------------------------------------------------
# Step 2: NVMMADistributedLayout declaration only — no MMA
# ---------------------------------------------------------------------------

@gluon.jit
def step2_layout(ptr, M: gl.constexpr, N: gl.constexpr):
    BLOCKED: gl.constexpr = gl.BlockedLayout([M * N // 128], [32], [4], [0])
    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, INSTR_SHAPE_N, 16],
    )
    acc = gl.zeros((M, N), dtype=gl.float32, layout=acc_layout)
    # Just pass through
    c = gl.load(ptr + gl.arange(0, M * N, layout=BLOCKED).to(gl.int64))
    gl.store(ptr + gl.arange(0, M * N, layout=BLOCKED).to(gl.int64), c)


def test_step2():
    C = torch.randn(M * N, device="cuda", dtype=torch.float32)
    step2_layout[(1,)](C, M=M, N=N, num_warps=4)
    print("Step 2 PASSED: NVMMADistributedLayout declaration works")


# ---------------------------------------------------------------------------
# Step 3: Full WGMMA (TMA + NVMMASharedLayout) — the real test
# ---------------------------------------------------------------------------

@gluon.jit
def step3_wgmma(a_desc, b_desc, c_desc, d_desc,
                INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr):
    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    # Allocate shared memory for A, B, C with NVMMASharedLayout (from TMA descriptors)
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    # TMA load with mbarrier
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_load(a_desc, [0, 0], bar, a_smem)
    tma.async_load(b_desc, [0, 0], bar, b_smem)
    tma.async_load(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Accumulator layout
    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    # Load C into registers with NVMMADistributedLayout
    c = c_smem.load(c_layout)

    # A is passed through shared memory (NVMMASharedLayout is already set by TMA)
    # B must be in shared memory for WGMMA
    a = a_smem

    # Issue async WGMMA: d = a * b + c
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

    # Store D back through TMA
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def test_step3():
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)

    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    step3_wgmma[(1,)](a_desc, b_desc, c_desc, d_desc,
                       INSTR_SHAPE_N=INSTR_SHAPE_N, num_warps=4)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)
    print("Step 3 PASSED: Full WGMMA with TMA works on this GPU!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    print(f"GPU capability: sm_{cap[0]}{cap[1]}")

    print("\n=== Step 1: Basic Gluon ===")
    test_step1()

    print("\n=== Step 2: NVMMADistributedLayout only ===")
    test_step2()

    print("\n=== Step 3: Full WGMMA (TMA + NVMMASharedLayout) ===")
    test_step3()

    print("\nAll steps passed!")