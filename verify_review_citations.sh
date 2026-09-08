#!/usr/bin/env bash
# verify_review_citations.sh -- runs INSIDE the dev container (pipe via docker exec -i bash -s)
set -u
V=/work/vllm-official
sep() { echo; echo "===== $1"; }

sep "tree present?"
ls -d $V/tests/models/test_marlin.py $V/tests/models/utils.py $V/tests/kernels/test_machete_gemm.py $V/tests/models/quantization/test_fp8.py 2>&1

sep "marlin: bitwise / top-3 / break at divergence / flaky reruns"
grep -n "bitwise\|top 3\|top-3\|Break out\|flaky\|reruns\|nondeterminism\|locks" $V/tests/models/test_marlin.py 2>/dev/null | head -20

sep "utils.check_logprobs_close signature + top-N + warn default"
grep -n "def check_logprobs_close" -A 14 $V/tests/models/utils.py 2>/dev/null | head -34
grep -n "must be in top\|warn_on_mismatch" $V/tests/models/utils.py 2>/dev/null | head -10

sep "fp8: only 4 tokens due to low-precision divergence"
grep -n "max_tokens\|low-precision\|numerical divergence" $V/tests/models/quantization/test_fp8.py 2>/dev/null | head -10

sep "machete tolerance + zeropoint note + cuda graph test"
grep -n "MNK_SHAPES" -A 8 $V/tests/kernels/test_machete_gemm.py 2>/dev/null | head -16
grep -n "atol\|rtol\|eropoint\|noise around" $V/tests/kernels/test_machete_gemm.py 2>/dev/null | head -16
grep -n "def test_machete_cuda_graph" $V/tests/kernels/test_machete_gemm.py 2>/dev/null

sep "logprobs cap (decides top-32 vs top-20 dump)"
grep -rn "max_num_logprobs\|MAX_LOGPROBS\|logprobs >" $V/vllm/sampling_params.py 2>/dev/null | head -12
grep -rn "max_num_logprobs" $V/vllm/entrypoints/*.py $V/vllm/entrypoints/**/*.py $V/vllm/config/*.py 2>/dev/null | head -12
grep -rn "prompt_logprobs" $V/vllm/entrypoints/openai/protocol.py 2>/dev/null | head -6
