#!/bin/bash
# Copy ALL TensileLibrary and gfx906 related files from backup directory
# This is the most likely fix for Tensile Segfault issue

set -e  # Exit on error

CONTAINER="vllm-gfx906"
BACKUP_DIR="/home/x/backup-rocm-tensors"
TENSILE_LIB_DIR="/usr/local/lib/python3.12/dist-packages/torch/lib/rocblas/library"

echo "=========================================="
echo "Copying ALL TensileLibrary Files for gfx906"
echo "=========================================="
echo ""

# Step 1: List all available files in backup directory
echo "[Step 1] Listing all files in backup directory..."
docker exec $CONTAINER bash -c "ls -lh $BACKUP_DIR/ | head -50"
echo ""

# Step 2: Count total files
echo "[Step 2] Counting files..."
TOTAL_FILES=$(docker exec $CONTAINER bash -c "ls $BACKUP_DIR/ | wc -l")
echo "Total files in backup directory: $TOTAL_FILES"
echo ""

# Step 3: List gfx906 specific files
echo "[Step 3] Listing gfx906-specific files..."
docker exec $CONTAINER bash -c "ls -lh $BACKUP_DIR/*gfx906* 2>/dev/null || echo 'No files with gfx906 in name'"
echo ""

# Step 4: List TensileLibrary files
echo "[Step 4] Listing TensileLibrary files..."
docker exec $CONTAINER bash -c "ls -lh $BACKUP_DIR/TensileLibrary* 2>/dev/null || echo 'No TensileLibrary files found'"
echo ""

# Step 5: Check current state of target directory
echo "[Step 5] Current Tensile files in container..."
docker exec $CONTAINER bash -c "ls -lh $TENSILE_LIB_DIR/TensileLibrary* 2>/dev/null || echo 'No TensileLibrary files in container yet'"
echo ""

# Step 6: Copy ALL .dat files (safest approach)
echo "[Step 6] Copying ALL .dat files from backup..."
docker exec $CONTAINER bash -c "
    cd $BACKUP_DIR
    echo 'Starting copy...'
    COPIED=0
    FAILED=0

    # Copy all .dat files
    for file in *.dat; do
        if [ -f \"\$file\" ]; then
            if cp -f \"\$file\" $TENSILE_LIB_DIR/; then
                echo \"✓ Copied: \$file\"
                COPIED=\$((COPIED + 1))
            else
                echo \"✗ Failed: \$file\"
                FAILED=\$((FAILED + 1))
            fi
        fi
    done

    echo ''
    echo 'Copy summary:'
    echo \"  Copied: \$COPIED files\"
    echo \"  Failed: \$FAILED files\"
"
echo ""

# Step 7: Copy .hsaco files if any (compiled HIP kernels)
echo "[Step 7] Checking for .hsaco files..."
docker exec $CONTAINER bash -c "
    cd $BACKUP_DIR
    HSACO_COUNT=\$(ls *.hsaco 2>/dev/null | wc -l)
    if [ \"\$HSACO_COUNT\" -gt 0 ]; then
        echo \"Found \$HSACO_COUNT .hsaco files, copying...\"
        for file in *.hsaco; do
            if [ -f \"\$file\" ]; then
                cp -f \"\$file\" $TENSILE_LIB_DIR/
                echo \"✓ Copied: \$file\"
            fi
        done
    else
        echo 'No .hsaco files found'
    fi
"
echo ""

# Step 8: Verify copied files
echo "[Step 8] Verifying copied files..."
docker exec $CONTAINER bash -c "
    echo 'Files in target directory:'
    ls -lh $TENSILE_LIB_DIR/*.dat 2>/dev/null | tail -20
    echo ''
    TOTAL_DAT=\$(ls $TENSILE_LIB_DIR/*.dat 2>/dev/null | wc -l)
    echo \"Total .dat files: \$TOTAL_DAT\"
"
echo ""

# Step 9: Set proper permissions
echo "[Step 9] Setting permissions..."
docker exec $CONTAINER bash -c "chmod 644 $TENSILE_LIB_DIR/*.dat 2>/dev/null && echo 'Permissions set'"
echo ""

# Step 10: Quick test with simple matmul
echo "[Step 10] Testing simple matmul..."
docker exec $CONTAINER bash -c 'python3 << "PYEOF"
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"HIP version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}")
print()

if not torch.cuda.is_available():
    print("ERROR: CUDA/HIP not available!")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

# Test 1: Simple matmul
print("Test 1: Simple 128x128 matmul...")
try:
    a = torch.randn(128, 128, dtype=torch.float16, device=device)
    b = torch.randn(128, 128, dtype=torch.float16, device=device)
    c = torch.mm(a, b)
    print(f"✓ PASSED - Result shape: {c.shape}")
except Exception as e:
    print(f"✗ FAILED - {type(e).__name__}: {e}")
    sys.exit(1)

# Test 2: Larger matmul (closer to model inference)
print("\nTest 2: Larger 1024x1024 matmul...")
try:
    a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    b = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    c = torch.mm(a, b)
    print(f"✓ PASSED - Result shape: {c.shape}")
except Exception as e:
    print(f"✗ FAILED - {type(e).__name__}: {e}")
    sys.exit(1)

# Test 3: Model-like linear layer
print("\nTest 3: Model-like linear layer (5120 -> 8192)...")
try:
    x = torch.randn(1, 5120, dtype=torch.float16, device=device)
    w = torch.randn(8192, 5120, dtype=torch.float16, device=device)
    y = torch.nn.functional.linear(x, w)
    print(f"✓ PASSED - Result shape: {y.shape}")
except Exception as e:
    print(f"✗ FAILED - {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All tests PASSED! ✓")
print("="*50)
PYEOF'
echo ""

# Step 11: Full vLLM test
echo "[Step 11] Running full vLLM test..."
docker exec $CONTAINER bash -c 'export HF_ENDPOINT=https://hf-mirror.com && python3 << "PYEOF"
from vllm import LLM, SamplingParams
import sys

print("Loading Qwen3-0.6B model...")
print()

try:
    llm = LLM(
        model="/home/x/models/vllm/Qwen3-0.6B-complete",
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        enforce_eager=False,  # Try with HIP Graph first
        max_model_len=256
    )

    print("Model loaded successfully!")
    print()

    print("Generating text...")
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0, max_tokens=20)
    )

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")

    print()
    print("="*50)
    print("✅ SUCCESS! Tensile Segfault is RESOLVED!")
    print("="*50)

except Exception as e:
    print(f"\n❌ FAILED: {type(e).__name__}: {str(e)[:500]}")
    print()
    print("If this is still Tensile Segfault, try:")
    print("1. Check if all files were copied correctly")
    print("2. Try setting ROCBLAS_TENSILE_USE_LAZY_LOADING=0")
    print("3. Use enforce_eager=True as temporary workaround")
    sys.exit(1)
PYEOF'
echo ""

echo "=========================================="
echo "All steps completed!"
echo "=========================================="
