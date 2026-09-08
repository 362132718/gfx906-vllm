#!/bin/bash
# Diagnose Tensile Segfault issue

echo "=== Step 1: Check TensileLibrary files ==="
docker exec vllm-gfx906 bash -c 'ls -lh /usr/local/lib/python3.12/dist-packages/torch/lib/rocblas/library/TensileLibrary*gfx906*'

echo ""
echo "=== Step 2: Check backup directory for more files ==="
docker exec vllm-gfx906 bash -c 'ls /home/x/backup-rocm-tensors/ | grep gfx906'

echo ""
echo "=== Step 3: Test with simple matmul ==="
docker exec vllm-gfx906 bash -c 'python3 << "PYEOF"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Current device: {torch.cuda.get_device_name(0)}")

    # Simple matmul test
    try:
        a = torch.randn(128, 128, dtype=torch.float16, device=device)
        b = torch.randn(128, 128, dtype=torch.float16, device=device)
        c = torch.mm(a, b)
        print(f"Simple matmul test: PASSED (result shape: {c.shape})")
    except Exception as e:
        print(f"Simple matmul test: FAILED - {e}")

    # Larger matmul test (similar to model inference)
    try:
        batch_size = 1
        seq_len = 10
        hidden_dim = 1024
        x = torch.randn(batch_size * seq_len, hidden_dim, dtype=torch.float16, device=device)
        w = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device=device)
        y = torch.mm(x, w)
        print(f"Model-like matmul test: PASSED (result shape: {y.shape})")
    except Exception as e:
        print(f"Model-like matmul test: FAILED - {e}")
PYEOF'

echo ""
echo "=== Step 4: Check rocBLAS environment variables ==="
docker exec vllm-gfx906 bash -c 'env | grep -i roc || echo "No ROCm env vars found"'

echo ""
echo "=== Step 5: Try disabling Tensile lazy loading ==="
docker exec vllm-gfx906 bash -c 'ROCBLAS_TENSILE_USE_LAZY_LOADING=0 python3 -c "import torch; a = torch.randn(10, 10, device=\"cuda\", dtype=torch.float16); print(torch.mm(a, a))"'
