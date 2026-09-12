#!/usr/bin/env bash
# p8_accept_27b.sh -- regression acceptance for the T1 model (Qwen3.8-27B dense)
# on the NEW delivery image.
#
# Why this is not optional: the previous :latest was built 39 h earlier from the
# STALE vllm-gfx906 container, so it never contained the P0 fix or the current
# GDN file. Today's image (c31a8b65) is the first carrying P0 + P3 + P0.5 + the
# m==1 branch + the tuned MoE config together. The 27B's 33.4 tok/s and image 6/6
# figures come from the C2 delivery of 2026-09-02 and therefore do NOT cover this
# image. Everything the 27B was promised (T1: load, text, image) is re-checked
# here using the image's own entrypoint at default settings.
set -uo pipefail

IMG=vllm-gfx906:latest
CTR=p8-27b
MODEL_DIR=/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
PORT=8172
WORK=/home/x/vllm-work/p8b
LOG="$WORK/p8_accept_27b.log"
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
say "--- gfx906-serve DEFAULT config, SERVED_NAME=qwen38 (a3_client hardcodes it)"
docker exec -d $CTR bash -lc \
  "cd /work && PORT=$PORT SERVED_NAME=qwen38 gfx906-serve $MODEL_DIR > /work/serve_27b.log 2>&1"
RC=1
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then RC=0; break; fi
  docker exec $CTR bash -lc 'pgrep -f "vllm serve" >/dev/null' || { RC=2; break; }
  sleep 10
done
say "serve RESULT=$([ $RC = 0 ] && echo STARTED || ([ $RC = 2 ] && echo CRASHED || echo TIMEOUT))"
if [ $RC -ne 0 ]; then
  docker exec $CTR bash -lc "tail -30 /work/serve_27b.log" 2>&1 | sed 's/^/    | /' | tee -a "$LOG"
  docker rm -f $CTR >/dev/null 2>&1
  say "P8_ACCEPT_27B_DONE rc=$RC"
  exit 1
fi
docker exec $CTR bash -lc "grep -aE 'GPU KV cache size|Using WNA16ExllamaKernel|Encoder cache' /work/serve_27b.log | tail -3" 2>&1 | sed 's/^/    | /' | tee -a "$LOG"

say "--- T1a decode speed (cold 1.6K prompt, warmup then REPS pair)"
docker exec $CTR bash -lc "python3 - <<'PY'
import json, time, urllib.request
base = 'Explain in detail why the sky is blue and what determines its shade at dawn. '
long_prompt = (base * 60)[:1600]
def call(ntok, text):
    p = {'model':'qwen38','max_tokens':ntok,'temperature':0,'ignore_eos':True,
         'messages':[{'role':'user','content':text}]}
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/chat/completions',
                                 data=json.dumps(p).encode(),
                                 headers={'Content-Type':'application/json'})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900))
    return time.time()-t0, d
call(8, long_prompt + ' warmup')                     # warmup, discarded
tA, dA = call(1,  long_prompt + ' Now answer in one word.')
tB, dB = call(64, long_prompt + ' Now answer in one word.')
n = dB['usage']['completion_tokens']
step = (tB - tA) / max(n - 1, 1)
print('T1A_PREFILL_1_SECONDS %.2f  T1B_64_SECONDS %.2f  n_out %d' % (tA, tB, n))
print('T1_DECODE_STEP_MS %.2f  T1_DECODE_TOK_S %.2f' % (step*1e3, 1/step))
PY" 2>&1 | tail -4 | sed 's/^/    | /' | tee -a "$LOG"

say "--- T1b text probe"
docker exec $CTR bash -lc "python3 - <<'PY'
import json, time, urllib.request
p = {'model':'qwen38','max_tokens':256,'temperature':0,
     'messages':[{'role':'user','content':'What is 17 times 23? Reply with only the number.'}]}
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/chat/completions',
                             data=json.dumps(p).encode(), headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req, timeout=600))
txt = d['choices'][0]['message']['content']
print('TEXT_SECONDS: %.1f' % (time.time()-t0))
print('TEXT_HAS_391: %s' % ('391' in txt))
PY" 2>&1 | tail -3 | sed 's/^/    | /' | tee -a "$LOG"

say "--- T1c image probe (a3_client.py)"
docker exec $CTR bash -lc "python3 -u /tmp/a3_client.py $PORT" 2>&1 | tail -5 | sed 's/^/    | /' | tee -a "$LOG"

say "--- T1d video probe, 8 frames"
docker exec $CTR bash -lc "python3 -u /tmp/a3_video_client.py $PORT qwen38 8" 2>&1 | tail -5 | sed 's/^/    | /' | tee -a "$LOG"

say "--- stopping"
docker exec $CTR bash -lc 'pkill -f "vllm serve"; sleep 6; pkill -9 -f "vllm serve"' >/dev/null 2>&1
sleep 4
docker rm -f $CTR >/dev/null 2>&1
say "P8_ACCEPT_27B_DONE rc=0"
