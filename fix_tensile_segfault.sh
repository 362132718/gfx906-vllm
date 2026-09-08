#!/bin/bash
# Complete fix script for Tensile Segfault issue
# This script will try multiple approaches in sequence

set -e  # Exit on error

CONTAINER="vllm-gfx906"
BACKUP_DIR="/home/x/backup-rocm-tensors"
TENSILE_LIB_DIR="/usr/local/lib/python3.12/dist-packages/torch/lib/rocblas/library"

echo "=========================================="
echo "Tensile Segfault Fix Script"
echo "=========================================="
echo ""

# Step 1: Check current state
echo "[Step 1] Checking current Tensile files..."
docker exec $CONTAINER bash -c "ls -lh $TENSILE_LIB_DIR/TensileLibrary*gfx906* 2>/dev/null || echo 'No gfx906 Tensile files found'"
echo ""

# Step 2: List all available backup files
echo "[Step 2] Listing backup files..."
docker exec $CONTAINER bash -c "find $BACKUP_DIR -name '*gfx906*' -o -name '*Tensile*' | head -30"
echo ""

# Step 3: Copy ALL Tensile files from backup (not just the main one)
echo "[Step 3] Copying ALL Tensile files from backup..."
docker exec $CONTAINER bash -c "
    cd $BACKUP_DIR
    # Copy all .dat files that might be related to gfx906
    for file in *.dat; do
        if [[ \$file == *'gfx906'* ]] || [[ \$file == *'TensileLibrary'* ]]; then
            cp -f \$file $TENSILE_LIB_DIR/
            echo \"Copied: \$file\"
        fi
    done
"
echo ""

# Step 4: Set proper permissions
echo "[Step 4] Setting permissions..."
docker exec $CONTAINER bash -c "chmod 644 $TENSILE_LIB_DIR/TensileLibrary*.dat 2>/dev/null || true"
echo ""

# Step 5: Test with simple matmul first
echo "[Step 5] Testing simple matmul..."
docker exec $CONTAINER bash -c 'python3 << "PYEOF"
import torch
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    try:
        a = torch.randn(128, 128, dtype=torch.float16, device=device)
        b = torch.randn(128, 128, dtype=torch.float16, device=device)
        c = torch.mm(a, b)
        print(f"✓ Simple matmul test PASSED")
    except Exception as e:
        print(f"✗ Simple matmul test FAILED: {e}")
        exit(1)
else:
    print("✗ CUDA not available")
    exit(1)
PYEOF'
echo ""

# Step 6: Test with vLLM using enforce_eager=True
echo "[Step 6] Testing vLLM with enforce_eager=True..."
docker exec $CONTAINER bash -c 'export HF_ENDPOINT=https://hf-mirror.com && python3 << "PYEOF"
from vllm import LLM, SamplingParams

try:
    llm = LLM(
        model="/home/x/models/vllm/Qwen3-0.6B-complete",
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=256
    )

    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0, max_tokens=10)
    )

    for output in outputs:
        print(f"✓ Generated: {output.outputs[0].text[:50]}...")

    print("✓ vLLM test with enforce_eager=True PASSED")
except Exception as e:
    print(f"✗ vLLM test FAILED: {e}")
    # Don't exit, try next step
PYEOF'
echo ""

# Step 7: Test without enforce_eager (original test)
echo "[Step 7] Testing vLLM without enforce_eager..."
docker exec $CONTAINER bash -c 'export HF_ENDPOINT=https://hf-mirror.com && timeout 60 python3 << "PYEOF" || echo "Test timed out or failed"
from vllm import LLM, SamplingParams

try:
    llm = LLM(
        model="/home/x/models/vllm/Qwen3-0.6B-complete",
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
        max_model_len=256
    )

    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0, max_tokens=10)
    )

    for output in outputs:
        print(f"✓ Generated: {output.outputs[0].text[:50]}...")

    print("✓ vLLM test without enforce_eager PASSED")
except Exception as e:
    print(f"✗ vLLM test without enforce_eager FAILED: {type(e).__name__}: {str(e)[:200]}")
PYEOF'
echo ""

echo "=========================================="
echo "Fix script completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If Step 5 failed: Need to check TensileLibrary file compatibility"
echo "2. If Step 6 passed but Step 7 failed: Use enforce_eager=True as workaround"
echo "3. If both Step 6 and 7 passed: Problem solved!"
echo ""
