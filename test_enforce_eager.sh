#!/bin/bash
# Test with enforce_eager=True to bypass HIP Graph/Tensile issues

docker exec vllm-gfx906 bash -c 'python3 << "PYEOF"
from vllm import LLM, SamplingParams

print("Testing with enforce_eager=True...")

llm = LLM(
    model="/home/x/models/vllm/Qwen3-0.6B-complete",
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    max_model_len=256
)

print("Model loaded successfully!")

outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0, max_tokens=20)
)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")

print("\nTest completed successfully!")
PYEOF'
