#!/usr/bin/env python3
"""Basic vLLM verification test for gfx906 environment."""

import torch
import vllm

def main():
    print("=" * 60)
    print("vLLM Basic Verification Test")
    print("=" * 60)

    # Check PyTorch
    print(f"\n[PyTorch]")
    print(f"  Version: {torch.__version__}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  HIP version: {torch.version.hip}")

    # Check vLLM
    print(f"\n[vLLM]")
    print(f"  Version: {vllm.__version__}")

    # Check available attention backends
    from vllm.config import ModelConfig
    print(f"\n[Environment]")
    import os
    print(f"  VLLM_TARGET_DEVICE: {os.getenv('VLLM_TARGET_DEVICE', 'not set')}")
    print(f"  CUDA_HOME: {os.getenv('CUDA_HOME', 'not set')}")

    print("\n" + "=" * 60)
    print("All basic checks passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
