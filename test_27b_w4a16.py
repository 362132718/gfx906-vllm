#!/usr/bin/env python3
"""
Test script for Qwen3.8-27B-W4A16 on gfx906 with TP2 and ExLlama kernel.

This script tests:
1. Model loading with compressed-tensors W4A16 quantization
2. TP2 (Tensor Parallel size 2) distribution
3. ExLlama GPTQ kernel execution
4. Text generation quality and determinism
5. Performance metrics

Usage:
    python3 test_27b_w4a16.py
"""

import os
import time
import sys

# Enable ExLlama WNA16 kernel
os.environ["VLLM_WNA16_EXLLAMA"] = "1"

from vllm import LLM, SamplingParams


def test_model_loading():
    """Test if the model loads correctly with W4A16 quantization."""
    print("=" * 70)
    print("TEST 1: Model Loading")
    print("=" * 70)

    try:
        llm = LLM(
            model="/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
            dtype="float16",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            max_model_len=256,
            attention_backend="TRITON_ATTN",
            disable_log_stats=False
        )
        print("✅ Model loaded successfully!")
        return llm
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None


def test_basic_generation(llm):
    """Test basic text generation."""
    print("\n" + "=" * 70)
    print("TEST 2: Basic Generation")
    print("=" * 70)

    prompt = "Hello, my name is"
    sampling_params = SamplingParams(temperature=0, max_tokens=30)

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start_time

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Time: {elapsed:.2f}s")

        # Check if output looks reasonable (not garbled)
        if len(generated_text.split()) > 5 and not all(c in '■□▪▫◆◇' for c in generated_text):
            print("✅ Output looks reasonable!")
            return True
        else:
            print("❌ Output appears garbled or too short")
            return False


def test_determinism(llm):
    """Test if generation is deterministic (same input produces same output)."""
    print("\n" + "=" * 70)
    print("TEST 3: Determinism Check")
    print("=" * 70)

    prompt = "The quick brown fox jumps over"
    sampling_params = SamplingParams(temperature=0, max_tokens=20)

    outputs1 = llm.generate([prompt], sampling_params)
    text1 = outputs1[0].outputs[0].text

    outputs2 = llm.generate([prompt], sampling_params)
    text2 = outputs2[0].outputs[0].text

    print(f"Run 1: {text1}")
    print(f"Run 2: {text2}")

    if text1 == text2:
        print("✅ Deterministic! Both runs produced identical output.")
        return True
    else:
        print("❌ Non-deterministic! Outputs differ between runs.")
        print("   This suggests tensor parallel partitioning issues.")
        return False


def test_multiple_prompts(llm):
    """Test generation with multiple different prompts."""
    print("\n" + "=" * 70)
    print("TEST 4: Multiple Prompts")
    print("=" * 70)

    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate factorial:",
        "What are the benefits of exercise?",
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=50, top_p=0.95)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    print(f"Generated {len(outputs)} responses in {elapsed:.2f}s\n")

    all_good = True
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        print(f"Prompt {i+1}: {prompts[i][:50]}...")
        print(f"Response: {generated[:100]}...")
        print()

        # Basic sanity check
        if len(generated.split()) < 10:
            print(f"  ⚠️ Response {i+1} seems too short")
            all_good = False

    if all_good:
        print("✅ All responses look reasonable!")
    else:
        print("⚠️ Some responses may have issues")

    return all_good


def test_performance(llm):
    """Measure performance metrics."""
    print("\n" + "=" * 70)
    print("TEST 5: Performance Metrics")
    print("=" * 70)

    prompt = "Write a detailed explanation of machine learning algorithms and their applications in modern technology."
    sampling_params = SamplingParams(temperature=0, max_tokens=100)

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start_time

    output_text = outputs[0].outputs[0].text
    output_tokens = len(output_text.split())
    input_tokens = len(prompt.split())

    input_speed = input_tokens / elapsed
    output_speed = output_tokens / elapsed

    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Input speed: {input_speed:.2f} toks/s")
    print(f"Output speed: {output_speed:.2f} toks/s")

    # Compare with baseline
    if output_speed > 30:
        print("✅ Excellent performance!")
    elif output_speed > 20:
        print("✅ Good performance")
    else:
        print("⚠️ Performance could be better")

    return output_speed


def main():
    print("\n" + "🧪" * 35)
    print("Testing Qwen3.8-27B-W4A16 on gfx906 with TP2")
    print("🧪" * 35 + "\n")

    # Test 1: Load model
    llm = test_model_loading()
    if not llm:
        print("\n❌ Cannot proceed without successful model loading")
        sys.exit(1)

    # Test 2: Basic generation
    basic_ok = test_basic_generation(llm)

    # Test 3: Determinism
    determinism_ok = test_determinism(llm)

    # Test 4: Multiple prompts
    multi_ok = test_multiple_prompts(llm)

    # Test 5: Performance
    perf_speed = test_performance(llm)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model Loading:      ✅ Success")
    print(f"Basic Generation:   {'✅' if basic_ok else '❌'} {'Pass' if basic_ok else 'Fail'}")
    print(f"Determinism:        {'✅' if determinism_ok else '❌'} {'Pass' if determinism_ok else 'Fail'}")
    print(f"Multiple Prompts:   {'✅' if multi_ok else '❌'} {'Pass' if multi_ok else 'Fail'}")
    print(f"Performance:        {perf_speed:.2f} toks/s")
    print("=" * 70)

    if basic_ok and determinism_ok:
        print("\n🎉 SUCCESS! W4A16 + TP2 + ExLlama working on gfx906!")
    else:
        print("\n⚠️ Issues detected. See details above.")
        if not determinism_ok:
            print("   → Likely tensor parallel partitioning issue in repack logic")
        if not basic_ok:
            print("   → Possible kernel execution or weight loading problem")


if __name__ == "__main__":
    main()
