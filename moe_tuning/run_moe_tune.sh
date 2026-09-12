#!/usr/bin/env bash
# run_moe_tune.sh -- run vLLM's upstream MoE Triton kernel tuner on the GPU host.
#
# Produces:  E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json
# loaded at serve time via  VLLM_TUNED_CONFIG_FOLDER=<dir>  -- no image rebuild.
#
# Why a SEPARATE container: the tuner needs `ray`, which is deliberately NOT in
# the delivery image. Only the resulting JSON ever ships, so the container is
# disposable and must not be the delivery candidate.
#
# Must run with nothing else on the GPU (the model is 24.3 GB over 2x16 GB).
#
# Usage:  bash run_moe_tune.sh [batch-sizes...]
#         bash run_moe_tune.sh 1 2 4 8 16 32 48
set -uo pipefail

DIR=/home/x/vllm-work/moe_tuning
CTR=p5-tune
IMAGE=vllm-gfx906:latest
MODEL=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
MODEL_HOST=/home/x/models
CFG_OUT="$DIR/configs"
BATCHES=("$@")
[ ${#BATCHES[@]} -eq 0 ] && BATCHES=(1 8)

mkdir -p "$CFG_OUT"
LOG="$DIR/tune.log"

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore|p5_.*\.py|llama-server" \
       | grep -v grep | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "TUNE_ABORT busy=$BUSY; stop everything else first" | tee -a "$LOG"
  ps -eo pid,etime,args | grep -E "[v]llm serve|[E]ngineCore|[p]5_" | cut -c1-120
  exit 1
fi

docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v "$MODEL_HOST":/models:ro -v "$DIR":/work \
  "$IMAGE" sleep infinity >/dev/null || { echo "TUNE_ABORT docker run failed" | tee -a "$LOG"; exit 1; }

echo "=== installing ray (dev-only, not in the delivery image) ===" | tee -a "$LOG"
docker exec "$CTR" bash -lc \
  'pip install --no-cache-dir --break-system-packages -q "ray[default]" 2>&1 | tail -3; python3 -c "import ray; print(\"ray\", ray.__version__)"' \
  | tee -a "$LOG"

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ | head -1)
CLK="$DIR/tune-clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "=== TUNE start $(date -Iseconds) batches='${BATCHES[*]}' ===" | tee -a "$LOG"
docker exec -d "$CTR" bash -lc "
  cd /work && export PYTHONNOUSERSITE=1 RAY_DISABLE_DASHBOARD=1 \
  VLLM_TUNED_CONFIG_FOLDER=/work/configs && \
  python3 -u /work/benchmark_moe.py \
    --model $MODEL \
    --tp-size 2 \
    --dtype int4_w4a16 \
    --trust-remote-code \
    --tune \
    --batch-size ${BATCHES[*]} \
    --save-dir /work/configs > /work/tune-run.log 2>&1"

for i in $(seq 1 480); do        # up to 2 h (docs say 30-90 min)
  sleep 15
  # tightened: the old pattern matched the bare word "Error" anywhere, which can
  # fire on benign log text and end the wait early. Match only terminal markers.
  if grep -qE "Writing best config|Tuning took|Traceback|ModuleNotFoundError" "$DIR/tune-run.log" 2>/dev/null; then
    sleep 30; break
  fi
  docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "!! container died" | tee -a "$LOG"; break; }
done

kill $SAMPLER 2>/dev/null; trap - EXIT
echo "=== TUNE done $(date -Iseconds) ===" | tee -a "$LOG"
tail -25 "$DIR/tune-run.log" 2>/dev/null | tee -a "$LOG"
echo "--- produced configs ---" | tee -a "$LOG"
ls -la "$CFG_OUT" | tee -a "$LOG"
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" | tee -a "$LOG"
echo "TUNE_ALL_DONE" | tee -a "$LOG"
