import re
from collections import defaultdict
import itertools
import math
from typing import List

def parse_layout(layout_str: str) -> tuple[dict, str, str]:
    """
    解析字符串格式的Triton布局。(已增强，可处理1D、2D等任意维度布局)

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
    # 增强后的正则表达式：捕获括号内的所有内容到 'coords' 组
    pattern = re.compile(r"^\s*(?:-\s*)?(?P<group>\w+)=\d+\s+->\s+\((?P<coords>.*?)\)")
    
    reps_line = ""
    out_dims_line = ""

    for line in layout_str.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = pattern.match(line)
        if match:
            group = match.group('group')
            coords_str = match.group('coords')
            
            # 将捕获的坐标字符串处理成整数元组，适用于 (32) 和 (0, 1) 等情况
            try:
                basis_vector = tuple(int(c.strip()) for c in coords_str.split(','))
            except ValueError:
                # 如果坐标字符串为空或格式不正确，则跳过
                continue

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
    """
    parsed_data, reps_line, out_dims_line = parse_layout(layout_str)

    vector_bases = parsed_data.get('vector', [])
    bank_bases = parsed_data.get('bank', [])
    segment_bases = parsed_data.get('segment', [])
    
    # --- 打印解析结果和警告的部分保持不变 ---
    print(f"原始布局解析结果:")
    print(f"- Vector bases: {vector_bases}")
    print(f"- Bank bases: {bank_bases} (数量: {len(bank_bases)})")
    print(f"- Segment bases: {segment_bases} (数量: {len(segment_bases)})")
    print("-" * 40)
    
    try:
        num_bank_perms = math.factorial(len(bank_bases))
        num_segment_perms = math.factorial(len(segment_bases))
        total_permutations = num_bank_perms * num_segment_perms
        print(f"理论上总布局组合数为 {total_permutations:,}。")
        print(f"将生成前 {max_permutations} 个组合。")
        print("=" * 40 + "\n")
    except ValueError:
        return []

    generated_layouts = []

    bank_perms_iter = itertools.permutations(bank_bases)
    segment_perms_iter = itertools.permutations(segment_bases)
    layout_iterator = itertools.product(bank_perms_iter, segment_perms_iter)

    for i, (permuted_bank, permuted_segment) in enumerate(layout_iterator):
        if i >= max_permutations:
            break

        current_layout_lines = []
        current_layout_lines.append(f"--- 布局组合 #{i + 1} ---")
        
        # 格式化输出函数，以正确显示元组
        def format_base(base_tuple):
            return f"({', '.join(map(str, base_tuple))})"

        if vector_bases:
            current_layout_lines.append(f"- vector=1 -> {format_base(vector_bases[0])}")
            
        for j, base in enumerate(list(permuted_bank)):
            prefix = f"- bank={2**j}" if j == 0 else f"  bank={2**j}"
            current_layout_lines.append(f"{prefix} -> {format_base(base)}")
            
        for j, base in enumerate(list(permuted_segment)):
            prefix = f"- segment={2**j}" if j == 0 else f"  segment={2**j}"
            current_layout_lines.append(f"{prefix} -> {format_base(base)}")
            
        if reps_line:
            current_layout_lines.append(f"- {reps_line}" if not reps_line.startswith('-') else reps_line)
        if out_dims_line:
            current_layout_lines.append(out_dims_line)
        
        generated_layouts.append("\n".join(current_layout_lines))

    return generated_layouts

def convert_to_triton_layout(layout_str: str) -> str:
    """
    Converts a human-readable, multi-line layout string into a compact,
    single-line format that is easier for C++ to parse.

    New Format Example:
    vector:1->(0,1),2->(8,0);bank:1->(0,8);reps:is a size 1 dimension
    """
    lines = layout_str.strip().split('\n')
    
    grouped_data = defaultdict(list)
    reps_line_content = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match "key=value -> (tuple)" format
        match = re.search(r'(\w+)=(\d+)\s*->\s*\((.*?)\)', line)
        if match:
            key, num, val_tuple = match.groups()
            # Create a sub-item like "1->(0,1)"
            sub_item = f"{num}->({val_tuple.replace(' ', '')})"
            grouped_data[key].append(sub_item)
        elif 'where out dims are:' in line:
            outdims = line.split(':')[1].strip()
            outdims = outdims.replace('[','').replace(']','').replace('size', '').replace('(', '').replace(')', '')
            outdims = outdims.replace('  ','->').replace(' ', '')
        elif 'rep' in line:
            # # For reps, just store the content
            # reps_line_content = line.lstrip('- ').strip()
            # reps_line_content = reps_line_content.replace('reps is a size', '').replace('dimension', '').strip()
            if 'is a size 1 dimension' in line:
                reps_line_content = '0'
    # Build the final string using the new format
    final_parts = []
    for key, sub_items in grouped_data.items():
        # Join all sub-items with a comma
        # e.g., "vector:1->(0,1),2->(8,0)"
        final_parts.append(f"{key}:{','.join(sub_items)}")
    
    if outdims:
        final_parts.append(f"outdims:{outdims}")
    
    if reps_line_content:
        # reps doesn't have a key-value structure, so we handle it differently
        # A simple way is to use a key like "reps"
        final_parts.append(f"reps:{reps_line_content}")
        
    # Join the final parts with a semicolon
    return ";".join(final_parts)


cpp_layout_str = """ 
 - vector=1 -> (32)
 - bank=1 -> (2)
   bank=2 -> (4)
   bank=4 -> (8)
   bank=8 -> (16)
   bank=16 -> (64)
 - segment=1 -> (65)
 - reps is a size 1 dimension
where out dims are: [dim0 (size 128)]
""" 

if __name__ == '__main__':
    res = get_layout_permutations(cpp_layout_str, max_permutations=10)
    for i, layout_str in enumerate(res):
        print(f"--- 列表元素 [{i}] ---")
        print(layout_str)
        if i < len(res) - 1:
            print("-" * 25)
         # Benchmark Triton implementation
        tmp_layout = convert_to_triton_layout(layout_str)
        print(tmp_layout)
    
    # triton_format_str = convert_to_triton_layout(cpp_layout_str)
    # print("Python script generated string (New Format):")
  
    print()