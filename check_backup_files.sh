#!/bin/bash
# Check what gfx906 files exist in backup directory and copy them if needed

echo "=== Listing all gfx906 files in backup directory ==="
docker exec vllm-gfx906 bash -c 'ls -lh /home/x/backup-rocm-tensors/*gfx906* 2>/dev/null || echo "No files found with gfx906 in name"'

echo ""
echo "=== Checking for TensileLibrary files ==="
docker exec vllm-gfx906 bash -c 'find /home/x/backup-rocm-tensors/ -name "*Tensile*" -type f | head -20'

echo ""
echo "=== Current Tensile files in container ==="
docker exec vllm-gfx906 bash -c 'ls -lh /usr/local/lib/python3.12/dist-packages/torch/lib/rocblas/library/TensileLibrary*'

echo ""
echo "=== File sizes comparison ==="
echo "Backup directory:"
docker exec vllm-gfx906 bash -c 'du -sh /home/x/backup-rocm-tensors/TensileLibrary* 2>/dev/null || echo "N/A"'

echo ""
echo "Container rocblas library:"
docker exec vllm-gfx906 bash -c 'du -sh /usr/local/lib/python3.12/dist-packages/torch/lib/rocblas/library/TensileLibrary*'
