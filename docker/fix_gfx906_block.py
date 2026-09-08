#!/usr/bin/env python3
"""Fix Triton decode attention BLOCK size for gfx906."""

import sys

def main():
    file_path = "/work/vllm-official/vllm/v1/attention/ops/triton_decode_attention.py"

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find and modify the BLOCK configuration
    modified = False
    for i, line in enumerate(lines):
        if line.strip() == "BLOCK = 32" and i > 500:
            # Check if next line is "if is_hip_:"
            if i + 1 < len(lines) and "if is_hip_:" in lines[i + 1]:
                # Add comment before
                lines[i] = "    # gfx906 (MI50/Vega20, wave64): BLOCK=32 is optimal per Z100 adaptation notes.\n"
                lines.insert(i + 1, "    # Other ROCm GPUs use BLOCK=16 for better occupancy.\n")
                lines.insert(i + 2, "    BLOCK = 32\n")
                # Modify the condition
                lines[i + 3] = "    if is_hip_ and not current_platform.is_gfx906():\n"
                lines[i + 4] = "        BLOCK = 16\n"
                modified = True
                print(f"Modified lines {i}-{i+4} for gfx906 BLOCK=32 optimization")
                break

    if not modified:
        print("ERROR: Could not find BLOCK configuration to modify", file=sys.stderr)
        sys.exit(1)

    # Add import if not present
    has_import = any("from vllm.platforms import current_platform" in line for line in lines[:50])
    if not has_import:
        # Find docstring end and add import
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') and i > 0:
                # Found closing docstring, add import after
                lines.insert(i + 1, "\nfrom vllm.platforms import current_platform\n")
                print("Added import for current_platform")
                break

    with open(file_path, "w") as f:
        f.writelines(lines)

    print("Successfully modified triton_decode_attention.py")

if __name__ == "__main__":
    main()
