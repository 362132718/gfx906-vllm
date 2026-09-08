#!/usr/bin/env bash
# p05_cold_serve.sh -- run ON THE HOST (x@192.168.10.11), detached, never in the
# foreground of an ssh pipe.
#
#   Phase 1  fresh container from the DELIVERY image (vllm-gfx906:latest) with an
#            empty /root/.cache -> cold start timing + serve-path tok/s at
#            DELIVERY CLOCKS (sclk auto/1700). Closes the two measurement gaps
#            P0.5 left open: the AOT invalidation caused by the patched
#            model_executor/layers/utils.py, and the ~36 tok/s that QUICKSTART
#            only extrapolated from the sclk ratio.
#   Phase 2  second start of the same container -> the cache-hit number
#            QUICKSTART promises (C2 measured 711 s cold / 420 s warm, pre-P0.5).
#   Phase 3  docker save + pigz -> rollback artifact, so the multi-day #20 kernel
#            work starts with a copy of the current shippable image that does not
#            live only on the machine with a silent power-loss history.
#
# Runs at whatever clocks the host is in (deliberately NOT pinned -- the whole
# point is the delivery clock). A /work/p0 bind mount is added beyond QUICKSTART's
# flags purely so the logs survive a host death; it cannot affect the compile
# cache, which lives in /root/.cache.
set -uo pipefail

IMAGE=vllm-gfx906:latest
CNAME=p05-cold
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
OUT=/home/x/vllm-work/p0
EXPORT_DIR=/home/x/image-export
PORT=8000
NAME=qwen38

boot0=$(uptime -s)
echo "=== P05 COLD/SERVE/EXPORT START $(date -Is) boot_since=$boot0"
docker image inspect "$IMAGE" --format 'image {{.Id}} size {{.Size}}'
df -h "$OUT" | tail -1

docker rm -f "$CNAME" >/dev/null 2>&1
echo "--- starting fresh container from $IMAGE"
docker run -d --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v /home/x/models:/home/x/models:ro \
  -v "$OUT":"$OUT" \
  -p $PORT:$PORT \
  "$IMAGE" sleep infinity >/dev/null || { echo "FATAL: docker run failed"; exit 1; }

echo "--- cache must be empty in a delivery container:"
docker exec "$CNAME" bash -c 'ls -a /root/.cache 2>&1 | head -3; du -sh /root/.cache 2>/dev/null || echo "(no /root/.cache)"'

# One serve cycle inside the container: $1 = tag (cold|warm), then model, port,
# served-name and the host log dir that is bind-mounted at the same path.
serve_once() {
  local tag="$1"
  docker exec -i "$CNAME" bash -s -- "$tag" "$MODEL" "$PORT" "$NAME" "$OUT" <<'INSIDE'
set -uo pipefail
tag=$1; MODEL=$2; PORT=$3; NAME=$4; OUT=$5
LOG=$OUT/p05_serve_$tag.log
echo "########## serve_once tag=$tag $(date -Is)"
echo "ENV: SKINNY=${VLLM_ROCM_USE_SKINNY_GEMM:-unset(=True)} cache_du=$(du -sh /root/.cache 2>/dev/null | cut -f1 || echo none)"

# A leftover worker from the previous phase would skew this one's timing AND OOM
# the new engine, so clear the pair before every cycle.
pgrep -f 'vllm serve|VLLM::Wor' | xargs -r kill -9 2>/dev/null
sleep 5
python3 -c 'import torch;print("preflight_vram_free_GB:",[round(torch.cuda.mem_get_info(i)[0]/1e9,1) for i in (0,1)])' 2>/dev/null || echo "preflight_vram_free_GB: unknown"

if [ "$tag" = cold ]; then
  gfx906-selfcheck > "$OUT/p05_selfcheck_fresh.log" 2>&1
  echo "SELFCHECK_RC=$? ok=$(grep -c '\[ok\]' "$OUT/p05_selfcheck_fresh.log") fail=$(grep -c FAIL "$OUT/p05_selfcheck_fresh.log")"
fi

gfx906-serve "$MODEL" > "$LOG" 2>&1 &
S=$!
T0=$(date +%s)
READY=0
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
  kill -0 $S 2>/dev/null || { echo "${tag}_DIED after $(( $(date +%s) - T0 ))s"; tail -40 "$LOG"; exit 1; }
  sleep 10
done
[ "$READY" = 1 ] || { echo "${tag}_TIMEOUT_1800s"; tail -40 "$LOG"; exit 1; }
echo "${tag}_READY_AFTER=$(( $(date +%s) - T0 ))s"
grep -a -oE "torch.compile took [0-9.]+ s" "$LOG" | head -1
grep -a -oE "Using (WNA16ExllamaKernel|TritonW4A16LinearKernel) for CompressedTensorsWNA16" "$LOG" | sort -u | head -2
grep -a -oE "GPU KV cache size: [0-9,]+ tokens.*" "$LOG" | head -1
grep -a -oE "\[serve\].*" "$LOG" | head -1
grep -a -ciE "skinny|LLGemm1" "$LOG" | sed 's/^/skinny_log_lines=/'

python3 - "$PORT" "$NAME" <<'PY'
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]

def ask(mt, prompt="The capital of France is"):
    body = json.dumps({"model": name, "prompt": prompt,
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1200))
    return d, time.time() - t0

# REQ1/REQ2 keep the C2 probe shape so 22.0 tok/s (pre-P0.5) stays comparable.
d, el = ask(24)
print("REQ1: %.1f tok/s %r" % (d["usage"]["completion_tokens"] / el, d["choices"][0]["text"][:40]))
d, el = ask(64)
print("REQ2_WARM: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
# The headline number: a decode long enough that prefill and first-token
# overhead stop mattering. Report every rep plus the best, never just the best.
rates = []
for i in range(3):
    d, el = ask(256, "请用一句话解释为什么显卡的显存带宽会限制大模型推理速度。")
    r = d["usage"]["completion_tokens"] / el
    rates.append(r)
    print("DECODE256 rep%d: %.1f tok/s  (%d tok / %.1fs) %r" % (
        i + 1, r, d["usage"]["completion_tokens"], el,
        d["choices"][0]["text"][:24]))
print("DECODE256 best=%.1f mean=%.1f" % (max(rates), sum(rates) / len(rates)))
PY

echo "--- cache footprint after $tag start:"
du -sh /root/.cache/* 2>/dev/null | head -5
kill $S 2>/dev/null; sleep 10; kill -9 $S 2>/dev/null
pgrep -f 'VLLM::Wor' | xargs -r kill -9 2>/dev/null
sleep 20
echo "remaining vllm procs: $(pgrep -cf 'vllm serve|VLLM::Wor' || echo 0)"
INSIDE
}

echo "=== PHASE 1 (cold) $(date -Is)"
serve_once cold

echo "--- thermal gap before the warm restart $(date -Is)"
sleep 180

echo "=== PHASE 2 (warm, same container, cache now populated) $(date -Is)"
serve_once warm

echo "=== PHASE 3 (export rollback artifact) $(date -Is)"
mkdir -p "$EXPORT_DIR"
TAR="$EXPORT_DIR/vllm-gfx906-1e18c8ea6b8b-p05.tar.gz"
df -h "$EXPORT_DIR" | tail -1
docker save "$IMAGE" | pigz -p 6 > "$TAR"
SAVE_RC=$?
echo "docker save|pigz rc=$SAVE_RC"
ls -l "$TAR" | awk '{printf "artifact_bytes=%s\n", $5}'
sha256sum "$TAR" | tee "$TAR.sha256"

echo "--- load-back proof (a rollback point that cannot be loaded is nothing)"
EXPECTED_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
docker load -i "$TAR" | tail -2
LOADED_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[ "$EXPECTED_ID" = "$LOADED_ID" ] && echo "LOADBACK_ID_MATCH $LOADED_ID" || echo "!! LOADBACK_ID_MISMATCH expected=$EXPECTED_ID loaded=$LOADED_ID"
df -h "$EXPORT_DIR" | tail -1

boot1=$(uptime -s)
echo "=== P05 COLD/SERVE/EXPORT DONE $(date -Is) boot_since=$boot1"
[ "$boot0" = "$boot1" ] && echo "NO_REBOOT_DURING_RUN" || echo "!! HOST_REBOOTED_MID_RUN (numbers invalid)"
echo P05_COLD_ALL_DONE
