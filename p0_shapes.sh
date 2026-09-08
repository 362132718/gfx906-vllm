#!/usr/bin/env bash
# p0_shapes.sh -- runs INSIDE the dev container, detached, with its own log:
#   docker exec -d vllm-gfx906 bash -c 'bash /work/p0/p0_shapes.sh > /work/p0/p0_shapes.out 2>&1'
# Starts the delivery entry point with the shape logger on, sends one batch-1
# request, shuts down gracefully (SIGINT so worker atexit handlers run), then
# prints the distinct (M,K,N) the W4A16 kernel actually received.
set -uo pipefail
P0=/work/p0
TAG=${TAG:-graph}                              # graph | eager
export VLLM_WNA16_EXL_SHAPEDBG=$P0/wna16_shapes.$TAG.csv
export PORT=8191 SERVED_NAME=qwen38
[ "${EAGER:-0}" = 1 ] && export EAGER=1
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
rm -f "$VLLM_WNA16_EXL_SHAPEDBG".* $P0/p0_serve.$TAG.log
echo "[p0_shapes] TAG=$TAG EAGER=${EAGER:-0}"

gfx906-serve "$MODEL" > $P0/p0_serve.$TAG.log 2>&1 &
SRV=$!
echo "[p0_shapes] serve pid=$SRV, waiting for startup"

ok=0
for i in $(seq 1 120); do          # up to 40 min; cold start measured at ~12 min
  sleep 20
  if grep -q "Application startup complete" $P0/p0_serve.$TAG.log 2>/dev/null; then ok=1; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[p0_shapes] serve died early"; break; fi
done
echo "[p0_shapes] startup_ok=$ok after $((i*20))s"
[ "$ok" != 1 ] && tail -25 $P0/p0_serve.$TAG.log

if [ "$ok" = 1 ]; then
  # 8 tokens of greedy decode: reaches every WNA16 projection at least once
  curl -s localhost:8191/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"qwen38","prompt":"Explain why the sky is blue at noon and red at dusk.","max_tokens":8,"temperature":0}' \
    | head -c 300
  echo
fi

# Flush every rank's CSV before teardown: vLLM hard-kills the TP workers, so
# atexit alone previously produced only the APIServer's empty file.
for p in $(pgrep -f 'vllm serve|EngineCore|VLLM::Wor'); do
  kill -USR1 "$p" 2>/dev/null && echo "[p0_shapes] USR1 -> $p"
done
sleep 3

kill -INT $SRV 2>/dev/null
for i in $(seq 1 12); do sleep 5; kill -0 $SRV 2>/dev/null || break; done
bash $P0/kill_serve.sh
sleep 3

echo "=== distinct (m,k,n) seen by the W4A16 kernel, per rank ==="
found=0
for f in "$VLLM_WNA16_EXL_SHAPEDBG".*; do
  [ -f "$f" ] || continue
  found=1
  echo "--- $f"
  tail -n +2 "$f" | sort | uniq -c | sort -k2
done
[ "$found" = 0 ] && echo "MISSING: no $VLLM_WNA16_EXL_SHAPEDBG.* (workers died before atexit)"
echo "=== graph config in force ==="
grep -oE '"cudagraph_mode":[^,}]*|cudagraph_capture_sizes[^,}]*' $P0/p0_serve.$TAG.log | sort -u | head -4
echo "Graph capturing finished count: $(grep -c 'Graph capturing finished' $P0/p0_serve.$TAG.log)"
grep -nE "Capturing CUDA graph|capturing finished|Dynamo bytecode transform time" $P0/p0_serve.$TAG.log | tail -5
echo "[p0_shapes] done"
