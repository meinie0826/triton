import re
from collections import defaultdict

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

if __name__ == '__main__':
    triton_format_str = convert_to_triton_layout(cpp_layout_str)
    print("Python script generated string (New Format):")
    print(triton_format_str)
    print()