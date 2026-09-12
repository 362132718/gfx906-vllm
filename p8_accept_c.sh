#!/usr/bin/env bash
# p8_accept_c.sh -- P8 acceptance part C: image + video + a readable text answer.
#
# Two earlier attempts failed on the served-model name, in opposite directions:
#   run 1: served name was the dir basename, payloads said "moe"      -> 404
#   run 2: served as "moe", but a3_client.py HARDCODES model=qwen38  -> 404 on image
# So serve as qwen38 (a3_client's hardcoded name) and pass qwen38 to the video
# client too. Video already scored 3/3 in run 2; it is repeated here so all three
# modalities are evidenced from ONE server instance.
#
# The text probe also uses max_tokens=256: at 64 the thinking preamble ate the
# whole budget and the answer never appeared (run 2 showed only "**Calculate:**").
set -uo pipefail

IMG=vllm-gfx906:latest
CTR=p8-accept
MODEL_DIR=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
PORT=8171
WORK=/home/x/vllm-work/p8
LOG="$WORK/p8_accept_c.log"
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

say "--- gfx906-serve DEFAULT config, SERVED_NAME=qwen38"
docker exec -d $CTR bash -lc \
  "cd /work && PORT=$PORT SERVED_NAME=qwen38 gfx906-serve $MODEL_DIR > /work/serve_c.log 2>&1"
RC=1
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then RC=0; break; fi
  docker exec $CTR bash -lc 'pgrep -f "vllm serve" >/dev/null' || { RC=2; break; }
  sleep 10
done
say "serve RESULT=$([ $RC = 0 ] && echo STARTED || ([ $RC = 2 ] && echo CRASHED || echo TIMEOUT))"
[ $RC -eq 0 ] || { docker exec $CTR bash -lc "tail -25 /work/serve_c.log"; exit 1; }
docker exec $CTR bash -lc "grep -aE 'GPU KV cache size' /work/serve_c.log | tail -1" | sed 's/^/    | /' | tee -a "$LOG"

say "--- C1 text probe (max_tokens=256, so the answer is visible)"
docker exec $CTR bash -lc "python3 - <<'PY'
import json, re, time, urllib.request
p = {'model':'qwen38','max_tokens':256,'temperature':0,
     'messages':[{'role':'user','content':'What is 17 times 23? Reply with only the number.'}]}
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/chat/completions',
                             data=json.dumps(p).encode(), headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req, timeout=600))
txt = d['choices'][0]['message']['content']
body = re.sub(r'<think>.*?</think>', '', txt, flags=re.S)
print('TEXT_SECONDS: %.1f' % (time.time()-t0))
print('TEXT_HAS_391: %s' % ('391' in txt))
print('TEXT_ANSWER: %r' % body.strip()[:120])
PY" 2>&1 | tail -4 | sed 's/^/    | /' | tee -a "$LOG"

say "--- C2 image probe (a3_client.py, hardcoded model=qwen38)"
docker exec $CTR bash -lc "python3 -u /tmp/a3_client.py $PORT" 2>&1 | tail -5 | sed 's/^/    | /' | tee -a "$LOG"

say "--- C3 video probe, 8 frames"
docker exec $CTR bash -lc "python3 -u /tmp/a3_video_client.py $PORT qwen38 8" 2>&1 | tail -6 | sed 's/^/    | /' | tee -a "$LOG"

say "--- stopping"
docker exec $CTR bash -lc 'pkill -f "vllm serve"; sleep 6; pkill -9 -f "vllm serve"' >/dev/null 2>&1
sleep 4
docker rm -f $CTR >/dev/null 2>&1
say "P8_ACCEPT_C_DONE"
