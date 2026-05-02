"""Dump ttgir for a warp-specialized matmul kernel to check MMA count."""
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip
from triton._C.libtriton import nvidia

triton.set_allocator(lambda s, a, st: torch.empty(s, dtype=torch.int8, device="cuda"))


@triton.jit
def check_kernel(a_ptr, b_ptr, c_ptr, s0, s1, s2, s3, s4, s5, M, N, K,
                 num_stages: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, GM: tl.constexpr, FP8: tl.constexpr,
                 ATMA: tl.constexpr, BTMA: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[s0, s1],
                                       block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[s2, s3],
                                       block_shape=[BN, BK])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[s4, s5],
                                       block_shape=[BM, BN])
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    group_size = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size
    k_tiles = tl.cdiv(K, BK)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=True, num_stages=num_stages):
        off_k = k * BK
        a = a_desc.load((pid_m * BM, off_k))
        b = b_desc.load((pid_n * BN, off_k))
        accumulator = tl.dot(a, b.T, accumulator)
    c = accumulator.to(tl.float16)
    c_desc.store((pid_m * BM, pid_n * BN), c)


M, N, K = 2048, 2048, 512
A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(N, K, dtype=torch.float16, device="cuda")
C = torch.empty(M, N, dtype=torch.float16, device="cuda")

grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128), )
k = check_kernel[grid](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                        M, N, K, 4, 128, 128, 64, 8, False, True, True)

ttgir = k.asm["ttgir"]
mma_count = ttgir.count("tc_gen5_mma")
has_ws = "ttg.warp_specialize" in ttgir

print(f"tc_gen5_mma count: {mma_count}")
print(f"Has warp_specialize: {has_ws}")

# Dump full ttgir to file
import sys
label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
outfile = f"ttgir_{label}.txt"
with open(outfile, "w") as f:
    f.write(f"tc_gen5_mma count: {mma_count}\n")
    f.write(f"Has warp_specialize: {has_ws}\n")
    f.write("\n=== TTGIR ===\n\n")
    f.write(ttgir)
print(f"Saved to {outfile}")