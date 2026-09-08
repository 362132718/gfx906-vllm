#!/usr/bin/env bash
# c1_cachekey_test.sh -- regression test for the compile-cache collision fix.
#
# PHASE 1: gfx906-serve with the ExLlama bridge (default). Cache root is now
#          mode-keyed, so this must reuse the warmed ~/.cache/vllm-w4a16-exllama
#          and come up fast with ~22 tok/s.
# PHASE 2: same serve with VLLM_WNA16_EXLLAMA=0. Before the fix this died at
#          profile time with "expected size 640==5120, stride 8192==1024"
#          because it loaded PHASE 1's artifacts (or vice versa). Now it gets
#          its own cache dir and must start cleanly on the Triton kernel.
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8122
export SERVED_NAME=qwen38

# Reuse the artifacts from the verified ExLlama run under the new mode-keyed name.
if [ ! -d "$HOME/.cache/vllm-w4a16-exllama" ] && [ -d "$HOME/.cache/vllm-c1exl" ]; then
  mv "$HOME/.cache/vllm-c1exl" "$HOME/.cache/vllm-w4a16-exllama"
fi

run_phase() {
  local label=$1 want_kernel=$2 log=$3
  echo "########## $label"
  gfx906-serve "$MODEL" > "$log" 2>&1 &
  local srv=$!
  local t0=$(date +%s) ready=0
  for i in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 $srv 2>/dev/null || { echo "${label}_SERVER_DIED"; grep -a -m3 -E "Error in op|expected size" "$log"; return 1; }
    sleep 10
  done
  [ "$ready" = 1 ] || { echo "${label}_TIMEOUT"; return 1; }
  echo "${label}_READY_AFTER=$(( $(date +%s) - t0 ))s"
  grep -a -oE "Using (WNA16ExllamaKernel|TritonW4A16LinearKernel) for CompressedTensorsWNA16" "$log" | sort -u | head -2
  grep -a -oE "\[serve\].*" "$log" | head -1
  if grep -aq "Using $want_kernel" "$log"; then echo "${label}_KERNEL=as_expected"; else echo "${label}_KERNEL=UNEXPECTED"; fi
  python3 - "$PORT" "$SERVED_NAME" <<'PY'
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
body = json.dumps({"model": name, "prompt": "The capital of France is",
                   "max_tokens": 48, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                            data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=600))
el = time.time() - t0
n = d["usage"]["completion_tokens"]
print("DECODE: %d tok in %.1fs = %.1f tok/s" % (n, el, n / el))
print("OUT:", repr(d["choices"][0]["text"][:60]))
PY
  kill $srv 2>/dev/null; sleep 8; kill -9 $srv 2>/dev/null
  bash /work/kill_serve.sh >/dev/null 2>&1
  sleep 6
  return 0
}

export VLLM_CACHE_ROOT="${HOME}/.cache/vllm"   # serve.sh appends -w4a16-<mode>
run_phase "PHASE1_EXLLAMA" "WNA16ExllamaKernel" /tmp/c1_p1.log
R1=$?
export VLLM_WNA16_EXLLAMA=0
run_phase "PHASE2_TRITON" "TritonW4A16LinearKernel" /tmp/c1_p2.log
R2=$?
echo "RESULT phase1_rc=$R1 phase2_rc=$R2"
ls -d "$HOME"/.cache/vllm-w4a16-* 2>/dev/null
du -sh "$HOME"/.cache/vllm-w4a16-* 2>/dev/null
echo C1_CACHEKEY_TEST_DONE
