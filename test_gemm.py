import sys
# --- 修改这里 ---
# 将下面的路径替换为你实际的 Triton 源码根目录路径
TRITON_SOURCE_ROOT = "/home/meiziyuan/triton/python"
# ---------------
# 将 Triton 源码根目录插入到 sys.path 的最前面
# 这样 Python 会优先在这个目录下查找 triton 包
sys.path.insert(0, TRITON_SOURCE_ROOT)

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

# The autotuner will try all configs and pick the best one.
# A good starting point are the configs from the Triton tutorial.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1 element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, masking out-of-bounds elements
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the pointers to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch grid to make autotuning work
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def benchmark_matmul(M, N, K, dtype=torch.float16, provider='triton'):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = do_bench(lambda: matmul(a, b), quantiles=quantiles)
    elif provider == 'torch':
        ms, min_ms, max_ms = do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        
    tflops = lambda ms: 2 * M * N * K / (ms * 1e-3) / 1e12
    
    return tflops(ms), tflops(max_ms), tflops(min_ms)

if __name__ == "__main__":
    # Define the matrix dimensions for benchmarking
    M, N, K = 1024, 3584, 18944
    DTYPE = torch.float16
    
    print(f"--- Benchmarking GEMM for {M}x{K}x{N} (dtype={DTYPE}) ---")
    
    # Benchmark Triton implementation
    triton_tflops, triton_max_tflops, triton_min_tflops = benchmark_matmul(M, N, K, DTYPE, provider='triton')
    print(f"Triton | Median TFLOPS: {triton_tflops:.2f} | Min TFLOPS: {triton_min_tflops:.2f} | Max TFLOPS: {triton_max_tflops:.2f}")
    
    # Benchmark PyTorch's native implementation
    torch_tflops, torch_max_tflops, torch_min_tflops = benchmark_matmul(M, N, K, DTYPE, provider='torch')
    print(f"PyTorch| Median TFLOPS: {torch_tflops:.2f} | Min TFLOPS: {torch_min_tflops:.2f} | Max TFLOPS: {torch_max_tflops:.2f}")

    # --- Verification ---
    print("\n--- Correctness Verification ---")
    a = torch.randn((M, K), device='cuda', dtype=DTYPE)
    b = torch.randn((K, N), device='cuda', dtype=DTYPE)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton's result is close to PyTorch's result.")
    else:
        print("❌ Verification FAILED: Triton's result differs from PyTorch's.")
        # Optional: print diff for debugging
        # print("Difference:", torch.max(torch.abs(triton_output - torch_output)))