#!/usr/bin/env bash
# p8_accept_b.sh -- P8 acceptance part B, rerun with the correct served-model
# name. The first run hardcoded "moe" in the payloads while gfx906-serve derives
# --served-model-name from the model directory basename, so every probe got
# HTTP 404 "The model `moe` does not exist". Step A (does the default config
# start?) already answered YES on the first pass and is not repeated here.
#
# Fix: pass SERVED_NAME=moe to gfx906-serve so the clients' payloads line up.
set -uo pipefail

IMG=vllm-gfx906:latest
CTR=p8-accept
MODEL_DIR=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
PORT=8171
WORK=/home/x/vllm-work/p8
LOG="$WORK/p8_accept_b.log"
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

say "--- starting gfx906-serve with DEFAULTS + SERVED_NAME=moe"
docker exec -d $CTR bash -lc \
  "cd /work && PORT=$PORT SERVED_NAME=moe gfx906-serve $MODEL_DIR > /work/serve_b.log 2>&1"
RC=1
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then RC=0; break; fi
  docker exec $CTR bash -lc 'pgrep -f "vllm serve" >/dev/null' || { RC=2; break; }
  sleep 10
done
say "serve RESULT=$([ $RC = 0 ] && echo STARTED || ([ $RC = 2 ] && echo CRASHED || echo TIMEOUT))"
[ $RC -eq 0 ] || { docker exec $CTR bash -lc "tail -20 /work/serve_b.log"; exit 1; }

say "--- served models"
docker exec $CTR bash -lc "curl -s http://127.0.0.1:$PORT/v1/models | head -c 300" | tee -a "$LOG"

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
PY" 2>&1 | tail -4 | sed 's/^/    | /' | tee -a "$LOG"

say "--- B2 image probe (a3_client.py)"
docker exec $CTR bash -lc "python3 -u /tmp/a3_client.py $PORT" 2>&1 | tail -4 | sed 's/^/    | /' | tee -a "$LOG"

say "--- B3 video probe, 8 frames"
docker exec $CTR bash -lc "python3 -u /tmp/a3_video_client.py $PORT moe 8" 2>&1 | tail -6 | sed 's/^/    | /' | tee -a "$LOG"

say "--- stopping"
docker exec $CTR bash -lc 'pkill -f "vllm serve"; sleep 6; pkill -9 -f "vllm serve"' >/dev/null 2>&1
sleep 4
docker rm -f $CTR >/dev/null 2>&1
say "P8_ACCEPT_B_DONE"
