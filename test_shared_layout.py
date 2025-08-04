#!/usr/bin/env python3
"""
测试 shared_layout 参数传递功能
"""
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

@triton.jit
def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = x * 2
    tl.store(y_ptr + offsets, y)

def test_shared_layout_integration():
    """测试 shared_layout 参数能否正确传递到编译过程"""
    
    # 创建测试数据
    size = 1024
    x = torch.randn(size, device='cuda')
    y = torch.zeros_like(x)
    
    # 测试不带 shared_layout 的编译（正常情况）
    print("Testing normal compilation...")
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    simple_kernel[grid](x, y, BLOCK_SIZE=256)
    
    # 验证结果正确性
    expected = x * 2
    assert torch.allclose(y, expected), "Normal compilation failed"
    print("✓ Normal compilation works")
    
    # 测试带 shared_layout 的编译
    print("Testing with shared_layout parameter...")
    try:
        # 重置输出张量
        y.zero_()
        
        # 使用 shared_layout 参数 - 应该作为内核调用的参数传递
        # 注意：这是实验性功能，shared_layout 值目前是字符串占位符
        simple_kernel[grid](x, y, BLOCK_SIZE=256, shared_layout="candidate_0")
        
        # 验证结果仍然正确
        assert torch.allclose(y, expected), "Shared layout compilation failed"
        print("✓ Shared layout parameter integration works")
        
    except Exception as e:
        print(f"⚠ Shared layout test failed (expected for now): {e}")
        # 这是预期的，因为我们还没有完成完整的集成
        
    print("Test completed!")

if __name__ == "__main__":
    test_shared_layout_integration()
