#!/usr/bin/env bash
# a2b_longctx.sh -- long-context DECODE measurement on the *delivery* path.
#
# Runs on the GPU host. Builds a brand-new container from vllm-gfx906:latest,
# starts it with the stock gfx906-serve entry point (nothing hand-installed),
# and probes decode tok/s at ~2K / ~8K / ~16K context.
#
# Why: the reference gfx906 fork reports that without a split-KV paged-decode
# route, 32K-context decode on MI50 collapses to ~0.24-1.35 tok/s, and that
# pure-Triton unified attention is ~16x slower than their SplitKV control at
# 32K per layer.  We have never measured long-context DECODE at all (only
# prefill), so #21 (hand-written attention) has no denominator.
#
# The host has a history of silent power loss, so everything runs detached and
# every step is written to /home/x/vllm-work/a2b/.
set -uo pipefail

DIR=/home/x/vllm-work/a2b
CTR=a2b-t1
IMAGE=vllm-gfx906:latest
MODEL_HOST=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
PORT=8117
TARGETS=${TARGETS:-2048,8192,16256}

mkdir -p "$DIR"
LOG="$DIR/driver.log"

# --- refuse to stack on a live engine (zombies do not hold the GPUs) ---------
BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore" | grep -v grep \
       | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "A2B_ABORT busy=$BUSY vllm processes still alive; not stacking a run" | tee -a "$LOG"
  ps -eo pid,stat,args | grep -E "vllm serve|EngineCore" | grep -v grep | head -5
  exit 1
fi
docker rm -f "$CTR" >/dev/null 2>&1 || true

# --- the in-container driver -------------------------------------------------
cat > "$DIR/inner.sh" <<INNER
#!/usr/bin/env bash
set -uo pipefail
echo "=== A2B LONG-CONTEXT DECODE PROBE \$(date -Iseconds) ==="
echo "--- fresh container from ${IMAGE}, stock gfx906-serve, targets=${TARGETS} ---"
export PORT=${PORT}
gfx906-serve /models/Qwen3.8-27B-OBLITERATED-AWQ-W4A16 > /work/serve.log 2>&1 &
SRV=\$!
echo "serve pid=\$SRV"
ready=""
for i in \$(seq 1 130); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; echo "READY_AFTER=\$((i*10))s"; break; fi
  if ! kill -0 \$SRV 2>/dev/null; then echo "SERVER_DIED"; tail -40 /work/serve.log; break; fi
  sleep 10
done
if [ -z "\$ready" ]; then echo "A2B_VERDICT=NO_START"; exit 1; fi
echo "--- engine config actually used ---"
grep -oE "Setting attention block size to [0-9]+ tokens" /work/serve.log | head -1
grep -oE "GPU KV cache size: [0-9,]+ tokens[^)]*\)" /work/serve.log | head -1
grep -oE "max_num_batched_tokens=[0-9]+" /work/serve.log | head -1
echo "--- probe ---"
MID=\$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "resolved_model_id=\$MID"
python3 -u /work/a2b_client.py ${PORT} "\$MID" "${TARGETS}"
RC=\$?
echo "--- post-probe engine scan ---"
grep -icE "traceback|CUDA error|illegal memory|out of memory|NaN" /work/serve.log | sed 's/^/fatal_signature_hits=/'
grep -icE "Prefix cache hit rate" /work/serve.log | sed 's/^/stat_lines=/'
grep -oE "Preempting [0-9]+ requests" /work/serve.log | head -3
kill \$SRV 2>/dev/null; wait \$SRV 2>/dev/null
echo "A2B_VERDICT=OK rc=\$RC"
echo "=== A2B DONE \$(date -Iseconds) ==="
INNER

cp "$(dirname "$0")/a2b_client.py" "$DIR/a2b_client.py" 2>/dev/null \
  || cp /home/x/vllm-work/a2b_client.py "$DIR/a2b_client.py" 2>/dev/null || true
if [ ! -s "$DIR/a2b_client.py" ]; then
  echo "A2B_ABORT a2b_client.py not staged in $DIR" | tee -a "$LOG"; exit 1
fi
bash -n "$DIR/inner.sh" || { echo "A2B_ABORT inner.sh syntax" | tee -a "$LOG"; exit 1; }

docker run -d --name "$CTR" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v "$MODEL_HOST/../":/models:ro \
  -v "$DIR":/work \
  "$IMAGE" sleep infinity >/dev/null || { echo "A2B_ABORT docker run failed" | tee -a "$LOG"; exit 1; }

echo "=== launching driver (detached) ===" | tee -a "$LOG"
setsid nohup docker exec "$CTR" bash /work/inner.sh >> "$LOG" 2>&1 < /dev/null &
echo "launcher pid=$!"
sleep 20
echo "--- $LOG so far ---"; tail -5 "$LOG"
