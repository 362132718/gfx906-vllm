#!/usr/bin/env bash
# p8_accept.sh -- P8 clean-container acceptance for the delivered image.
#
# Two questions, in order:
#   A) does `gfx906-serve` come up on the MoE model with its OWN DEFAULTS?
#      (serve.sh defaults MAX_NUM_SEQS=48 / GPU_MEM_UTIL=0.90; a hybrid GDN model
#       needs one Mamba cache block per sequence, so this may refuse to start)
#   B) with the known-good MAX_NUM_SEQS=32, do the three advertised modalities
#      actually work out of the box -- text, image, video?
#
# Runs everything INSIDE a fresh container from :latest using the image's own
# entrypoint (gfx906-serve), not our test harness. That is what "clean-machine
# acceptance" has to mean for this project.
set -uo pipefail

IMG=vllm-gfx906:latest
CTR=p8-accept
MODEL_DIR=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
PORT=8171
WORK=/home/x/vllm-work/p8
LOG="$WORK/p8_accept.log"
mkdir -p "$WORK"
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

docker rm -f $CTR >/dev/null 2>&1
docker run -d --name $CTR \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v /home/x/models:/models:ro -v "$WORK":/work \
  -p $PORT:$PORT "$IMG" sleep infinity >/dev/null || { say "docker run FAILED"; exit 1; }
sleep 6
for f in a3_client.py a3_video_client.py mkvideo.py; do
  docker cp "/home/x/vllm-work/p8/$f" "$CTR:/tmp/$f" >/dev/null
done
say "container $CTR up from $IMG"
docker exec $CTR python3 -c "import PIL, cv2; print('  deps: PIL', PIL.__version__, '| cv2', cv2.__version__)" 2>&1 | tee -a "$LOG"

wait_health() {   # $1 = timeout sec, returns 0 healthy / 2 crashed / 1 timeout
  local t=${1:-900} i
  for i in $(seq 1 $((t / 10))); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    docker exec $CTR bash -lc 'pgrep -f "vllm serve" >/dev/null' || return 2
    sleep 10
  done
  return 1
}

serve_start() {   # $1 = label, rest = extra env assignments
  local label=$1; shift
  local envs="$*"
  say "--- serve_start $label  env: $envs"
  docker exec -d $CTR bash -lc "cd /work && PORT=$PORT $envs gfx906-serve $MODEL_DIR > /work/serve_$label.log 2>&1"
  wait_health 900
  local rc=$?
  case $rc in
    0) say "    ${label}_RESULT=STARTED";;
    2) say "    ${label}_RESULT=CRASHED";;
    *) say "    ${label}_RESULT=TIMEOUT_900s";;
  esac
  local hint
  hint=$(docker exec $CTR bash -lc "grep -aoE 'GPU KV cache size: [0-9,]+ tokens|available Mamba cache blocks \([0-9]+\)|max_num_seqs \([0-9]+\) exceeds[^\"]{0,60}' /work/serve_$label.log | tail -2" 2>/dev/null | tr '\n' ' ')
  say "    ${label}_DIAG: $hint"
  return $rc
}

serve_stop() {
  docker exec $CTR bash -lc 'pkill -f "vllm serve"; sleep 6; pkill -9 -f "vllm serve"' >/dev/null 2>&1
  sleep 6
  docker restart $CTR >/dev/null 2>&1     # also frees the GPU context
  sleep 6
}

# ---------------- A: defaults ----------------
say "########## A: gfx906-serve with its own defaults"
serve_start defaults
A_RC=$?
docker exec $CTR bash -lc "grep -aE 'max_num_seqs|Mamba cache blocks|KV cache size|ValueError' /work/serve_defaults.log | tail -4" 2>&1 | sed 's/^/    | /' | tee -a "$LOG"
serve_stop

# ---------------- B: known-good MAX_NUM_SEQS=32 ----------------
say "########## B: MAX_NUM_SEQS=32 -> text + image + video"
serve_start cn32 MAX_NUM_SEQS=32
B_RC=$?
if [ $B_RC -eq 0 ]; then
  docker exec $CTR bash -lc "grep -aE 'GPU KV cache size|Mamba cache blocks' /work/serve_cn32.log | tail -2" 2>&1 | sed 's/^/    | /' | tee -a "$LOG"

  say "--- B1 text probe"
  docker exec $CTR bash -lc "python3 - <<'PY'
import json, time, urllib.request
p = {'model':'moe','max_tokens':64,'temperature':0,
     'messages':[{'role':'user','content':'What is 17 times 23? Reply with only the number.'}]}
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/chat/completions',
                             data=json.dumps(p).encode(), headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req, timeout=600))
print('TEXT_SECONDS: %.1f' % (time.time()-t0))
print('TEXT_RESPONSE: %r' % d['choices'][0]['message']['content'][:160])
PY" 2>&1 | sed 's/^/    | /' | tee -a "$LOG"

  say "--- B2 image probe (a3_client.py)"
  docker exec $CTR bash -lc "python3 -u /tmp/a3_client.py $PORT" 2>&1 | tail -4 | sed 's/^/    | /' | tee -a "$LOG"

  say "--- B3 video probe, 8 frames (a3_video_client.py)"
  docker exec $CTR bash -lc "python3 -u /tmp/a3_video_client.py $PORT moe 8" 2>&1 | tail -5 | sed 's/^/    | /' | tee -a "$LOG"
else
  say "    (skipped: server did not start)"
  docker exec $CTR bash -lc "grep -aE 'max_num_seqs|Mamba|ValueError|Traceback' /work/serve_cn32.log | tail -4" 2>&1 | sed 's/^/    | /' | tee -a "$LOG"
fi
serve_stop

say "########## done; raw logs under $WORK"
say "P8_ACCEPT_DONE A_RC=$A_RC B_RC=$B_RC"
