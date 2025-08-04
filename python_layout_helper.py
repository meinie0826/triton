import re

def convert_to_triton_layout(layout_str):
    """
    Converts a human-readable layout string to a format that can be parsed by Triton's C++ code.
    """
    # Clean the input string
    lines = layout_str.strip().split('\n')
    # Filter out irrelevant lines and clean up whitespace
    cleaned_lines = [re.sub(r'^\s*-\s*', '', line).strip() for line in lines if '->' in line or 'is a size' in line]
    
    # Group lines by dimension (vector, bank, etc.)
    grouped_parts = {}
    current_key = None
    for line in cleaned_lines:
        match = re.match(r'(\w+)=', line)
        if match:
            current_key = match.group(1)
            if current_key not in grouped_parts:
                grouped_parts[current_key] = []
            grouped_parts[current_key].append(line)
        elif current_key and '->' in line: # Handle lines that continue a dimension
             # This case might not be strictly needed with the new logic but is good for robustness
            grouped_parts[current_key].append(f"{current_key}={line}")
        elif 'is a size' in line:
            grouped_parts['reps'] = [line]

    # Format into the final string
    items = []
    for key, lines in grouped_parts.items():
        if key in ['vector', 'bank', 'segment']:
            sub_items = []
            for line in lines:
                # Extract number and tuple part, e.g., from "vector=1 -> (0, 1)"
                match = re.search(r'(\d+)\s*->\s*\(([^)]+)\)', line)
                if match:
                    num, val = match.groups()
                    # Convert (x, y) to {x,y}
                    sub_value = f"{{{val.replace(' ', '')}}}"
                    sub_items.append(f"{num}->{sub_value}")
            items.append(f"{key}={{{','.join(sub_items)}}}")
        elif key == 'reps':
            items.append("reps is a size 1 dimension")

    return f"{{{','.join(items)}}}"

if __name__ == '__main__':
    # Example usage with the layout string you provided
    cpp_output_str = """
     - vector=1 -> (0, 1)
       vector=2 -> (8, 0)
     - bank=1 -> (0, 8)
       bank=2 -> (0, 16)
       bank=4 -> (0, 32)
       bank=8 -> (0, 64)
     - segment=1 -> (4, 0)
       segment=2 -> (16, 0)
       segment=4 -> (32, 0)
       segment=8 -> (64, 0)
       segment=16 -> (0, 128)
       segment=32 -> (1, 8)
       segment=64 -> (2, 16)
       segment=128 -> (0, 34)
       segment=256 -> (0, 68)
     - reps is a size 1 dimension
    """

    triton_format_str = convert_to_triton_layout(cpp_output_str)
    print("Triton-parsable layout string:")
    print(triton_format_str)

    # Expected output format for C++ fromString:
    # {vector={1->{0,1},2->{8,0}},bank={1->{0,8},2->{0,16},4->{0,32},8->{0,64}},segment={1->{4,0},2->{16,0},4->{32,0},8->{64,0},16->{0,128},32->{1,8},64->{2,16},128->{0,34},256->{0,68}},reps is a size 1 dimension}
    
    # You would then pass this string to the Triton kernel launch
    # e.g., kernel.run(..., shared_layout=triton_format_str)

