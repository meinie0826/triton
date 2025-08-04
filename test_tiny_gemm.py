import sys
# --- 修改这里 ---
# 将下面的路径替换为你实际的 Triton 源码根目录路径
TRITON_SOURCE_ROOT = "/home/meiziyuan/triton/python"
# ---------------
# 将 Triton 源码根目录插入到 sys.path 的最前面
# 这样 Python 会优先在这个目录下查找 triton 包
sys.path.insert(0, TRITON_SOURCE_ROOT)

# 现在导入的就是你指定路径下的 Triton 了
import triton
from triton import language as tl
import torch
from python_layout_helper import convert_to_triton_layout

@triton.jit
def tiny_matmul_kernel(a_ptr, b_ptr, c_ptr):
    offs = tl.arange(0, 16)
    a = tl.load(a_ptr + offs[:, None] * 16 + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * 16 + offs[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * 16 + offs[None, :], c)

# 使用示例
a = torch.randn(16, 16, device='cuda', dtype=torch.float16)
b = torch.randn(16, 16, device='cuda', dtype=torch.float16)
c = torch.empty(16, 16, device='cuda', dtype=torch.float16)

# 1. 从 C++ 输出获取
cpp_layout_str = """ 
 - vector=1 -> (0, 1)
 - bank=1 -> (1, 0)
   bank=2 -> (2, 0)
   bank=4 -> (0, 2)
   bank=8 -> (0, 4)
   bank=16 -> (0, 8)
 - segment=1 -> (8, 0)
   segment=2 -> (4, 8)
 - reps is a size 1 dimension
where out dims are: [dim0 (size 16), dim1 (size 16)]
""" 
# 2. 转换
parsable_layout_str = convert_to_triton_layout(cpp_layout_str)
# 3. 编译时传入
tiny_matmul_kernel[(1,)](a, b, c, shared_layout=parsable_layout_str)
ref = torch.matmul(a, b)
torch.testing.assert_close(c, ref)
print("Test passed!")