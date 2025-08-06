"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Optional
import torch

import triton
import triton.language as tl
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(group_A, group_B, shared_layout):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        shared_layout=shared_layout,
    )

    return group_C


group_m = [1024]
group_n = [1024]
group_k = [1024]
group_A = []
group_B = []
group_B_T = []
assert len(group_m) == len(group_n)
assert len(group_n) == len(group_k)
group_size = len(group_m)
for i in range(group_size):
    M = group_m[i]
    N = group_n[i]
    K = group_k[i]
    A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
    B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
    B_T = B.T.contiguous()
    group_A.append(A)
    group_B.append(B)
    group_B_T.append(B_T)

layout = """
 - vector=1 -> (8, 0)
 - bank=1 -> (0, 1)
   bank=2 -> (0, 8)
   bank=4 -> (0, 16)
   bank=8 -> (0, 2)
   bank=16 -> (0, 4)
 - segment=1 -> (16, 0)
   segment=2 -> (32, 0)
   segment=4 -> (0, 32)
   segment=8 -> (0, 64)
   segment=16 -> (1, 1)
   segment=32 -> (2, 8)
   segment=64 -> (4, 16)
 - reps is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 128)]
"""
from python_layout_helper import convert_to_triton_layout
# layout = convert_to_triton_layout(layout)
# tri_out = group_gemm_fn(group_A, group_B,shared_layout=layout)
# ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
# for i in range(group_size):
#     assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2)


def benchmark_matmul(M, N, K, dtype=torch.float16, provider='triton', layout=layout):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = do_bench(lambda: group_gemm_fn([a], [b], shared_layout=layout), quantiles=quantiles)
    elif provider == 'torch':
        ms, min_ms, max_ms = do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        
    tflops = lambda ms: 2 * M * N * K / (ms * 1e-3) / 1e12
    
    return tflops(ms), tflops(max_ms), tflops(min_ms)

import re
import itertools
import math
from typing import List

def parse_layout(layout_str: str) -> tuple[dict, str, str]:
    """
    解析字符串格式的Triton布局。(已修正正则表达式)

    Args:
        layout_str: 包含布局信息的字符串。

    Returns:
        一个元组，包含：
        - 一个字典，存储了'vector', 'bank', 'segment'的基向量列表。
        - 'reps'行的字符串。
        - 'out dims'行的字符串。
    """
    parsed_data = {
        'vector': [],
        'bank': [],
        'segment': []
    }
    pattern = re.compile(r"^\s*(?:-\s*)?(?P<group>\w+)=\d+\s+->\s+\((?P<d0>\d+),\s*(?P<d1>\d+)\)")
    
    reps_line = ""
    out_dims_line = ""

    for line in layout_str.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = pattern.match(line)
        if match:
            group = match.group('group')
            basis_vector = (int(match.group('d0')), int(match.group('d1')))
            if group in parsed_data:
                parsed_data[group].append(basis_vector)
        elif 'reps is a' in line:
            reps_line = line
        elif 'out dims are:' in line:
            out_dims_line = line
            
    return parsed_data, reps_line, out_dims_line

def get_layout_permutations(layout_str: str, max_permutations: int = 10) -> List[str]:
    """
    根据给定的布局字符串，生成所有可能的布局排列，并以字符串列表形式返回。
    排列仅在 'bank' 和 'segment' 分组内部进行。

    Args:
        layout_str: 包含布局信息的字符串。
        max_permutations: 最多生成的排列数量。

    Returns:
        一个字符串列表，每个字符串是一个完整的布局排列。
    """
    # 1. 解析输入字符串
    parsed_data, reps_line, out_dims_line = parse_layout(layout_str)

    vector_bases = parsed_data.get('vector', [])
    bank_bases = parsed_data.get('bank', [])
    segment_bases = parsed_data.get('segment', [])

    print(f"原始布局解析结果:")
    print(f"- Vector bases: {vector_bases}")
    print(f"- Bank bases: {bank_bases} (数量: {len(bank_bases)})")
    print(f"- Segment bases: {segment_bases} (数量: {len(segment_bases)})")
    print("-" * 40)
    
    # ... (计算总数和打印警告的部分保持不变)
    try:
        num_bank_perms = math.factorial(len(bank_bases))
        num_segment_perms = math.factorial(len(segment_bases))
        total_permutations = num_bank_perms * num_segment_perms
        print(f"理论上总布局组合数为 {total_permutations:,}。")
        print(f"将生成前 {max_permutations} 个组合。")
        print("=" * 40 + "\n")
    except ValueError:
        return []

    # 2. 创建一个列表来存储结果
    generated_layouts = []

    # 3. 创建迭代器
    bank_perms_iter = itertools.permutations(bank_bases)
    segment_perms_iter = itertools.permutations(segment_bases)
    layout_iterator = itertools.product(bank_perms_iter, segment_perms_iter)

    # 4. 循环生成布局，并存入列表
    for i, (permuted_bank, permuted_segment) in enumerate(layout_iterator):
        # if i >= max_permutations:
        #     break

        # 创建一个临时列表来构建当前布局的每一行
        current_layout_lines = []
        
        current_layout_lines.append(f"--- 布局组合 #{i + 1} ---")
        
        if vector_bases:
            current_layout_lines.append(f"- vector=1 -> {vector_bases[0]}")
            
        for j, base in enumerate(list(permuted_bank)):
            prefix = f"- bank={2**j}" if j == 0 else f"  bank={2**j}"
            current_layout_lines.append(f"{prefix} -> {base}")
            
        for j, base in enumerate(list(permuted_segment)):
            prefix = f"- segment={2**j}" if j == 0 else f"  segment={2**j}"
            current_layout_lines.append(f"{prefix} -> {base}")
            
        if reps_line:
            current_layout_lines.append(f"- {reps_line}" if not reps_line.startswith('-') else reps_line)
        if out_dims_line:
            current_layout_lines.append(out_dims_line)
        
        # 将所有行合并成一个字符串，并添加到结果列表中
        generated_layouts.append("\n".join(current_layout_lines))

    # 5. 返回结果列表
    return generated_layouts

if __name__ == "__main__":
    # Define the matrix dimensions for benchmarking
    M, N, K = 1024, 1024, 1024
    DTYPE = torch.float16
    # Benchmark PyTorch's native implementation
    print(f"--- Benchmarking GEMM for {M}x{K}x{N} (dtype={DTYPE}) ---")
    torch_tflops, torch_max_tflops, torch_min_tflops = benchmark_matmul(M, N, K, DTYPE, provider='torch')
    print(f"PyTorch| Median TFLOPS: {torch_tflops:.2f} | Min TFLOPS: {torch_min_tflops:.2f} | Max TFLOPS: {torch_max_tflops:.2f}")

    list_of_layouts = get_layout_permutations(layout, max_permutations=5)
    triton_results = []
    # 打印返回的列表中的每一个布局
    print(f"函数返回了 {len(list_of_layouts)} 个布局。")
    print("--- 开始打印返回的列表内容 ---")
    for i, layout_str in enumerate(list_of_layouts):
        print(f"--- 列表元素 [{i}] ---")
        print(layout_str)
        if i < len(list_of_layouts) - 1:
            print("-" * 25)
         # Benchmark Triton implementation
        tmp_layout = convert_to_triton_layout(layout_str)
        triton_tflops, triton_max_tflops, triton_min_tflops = benchmark_matmul(M, N, K, DTYPE, provider='triton', layout=tmp_layout)
        print(f"Triton | Median TFLOPS: {triton_tflops:.2f} | Min TFLOPS: {triton_min_tflops:.2f} | Max TFLOPS: {triton_max_tflops:.2f}")
        # 记录结果：(tflops, layout, layout_str)
        triton_results.append((triton_tflops, tmp_layout, layout_str, i))

    # 根据 triton_tflops 排序（降序）
    triton_results.sort(key=lambda x: x[0], reverse=True)

    # 打印排序后的结果
    print("\n" + "="*50)
    print("性能排序结果（按 TFLOPS 降序排列）:")
    print("="*50)

    for i, (tflops, layout, layout_str, original_index) in enumerate(triton_results):
        print(f"排名 {i+1}: TFLOPS = {tflops:.2f}")
        print(f"  原始索引: {original_index}")
        print(f"  布局内容: {layout_str}")
        print("-" * 30)

    # 如果需要获取最佳布局
    if triton_results:
        best_tflops, best_layout, best_layout_str, best_index = triton_results[0]
        print(f"\n🏆 最佳性能: {best_tflops:.2f} TFLOPS")
        print(f"   对应布局索引: {best_index}")
        print(f"   对应布局内容: {best_layout_str}")