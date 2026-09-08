#!/usr/bin/env bash
# c3_video_verify.sh -- decide two things with evidence, eager startup:
#   A) does video:1 with the model's own size budget (16777216) actually die?
#      (previous run only hit our 6-minute wait window, which is not proof)
#   B) with the capped size, does a real video request work, and does the
#      image path still work in the same server?
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8161
export EAGER=1
export MM='{"image":1,"video":1}'

serve_wait() {  # label, extra args...
  local label=$1 log=/tmp/vv_$1.log
  shift
  gfx906-serve "$MODEL" "$@" > "$log" 2>&1 &
  local srv=$! i rc=1
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then rc=0; break; fi
    kill -0 $srv 2>/dev/null || { rc=2; break; }
    sleep 10
  done
  case $rc in
    0) echo "${label}_RESULT=STARTED after ~$((i*10))s";;
    2) echo "${label}_RESULT=CRASHED after ~$((i*10))s";;
    *) echo "${label}_RESULT=STILL_RUNNING_after_900s";;
  esac
  VV_SRV=$srv
  VV_RC=$rc
  VV_LOG=$log
}

stop_serve() {
  kill $VV_SRV 2>/dev/null; sleep 8; kill -9 $VV_SRV 2>/dev/null
  bash /work/kill_serve.sh >/dev/null 2>&1
  sleep 6
}

echo "########## A: model default size budget (16777216)"
serve_wait baseline
grep -a -oE "Encoder cache will be initialized with a budget of [0-9]+ tokens[^\"]*" "$VV_LOG" | tail -1
grep -a -oE "Tried to allocate [0-9.]+ [KMGT]iB[^\"]*" "$VV_LOG" | tail -2
grep -a -oE "torch\.OutOfMemoryError[^\"]{0,80}" "$VV_LOG" | tail -1
stop_serve

echo "########## B: capped size 4194304"
serve_wait capped --mm-processor-kwargs '{"size": {"longest_edge": 4194304, "shortest_edge": 65536}}'
if [ "$VV_RC" = 0 ]; then
  grep -a -oE "Encoder cache will be initialized with a budget of [0-9]+ tokens[^\"]*" "$VV_LOG" | tail -1
  echo "--- image probe (regression) ---"
  python3 -u /tmp/a3_client.py "$PORT" 2>&1 | grep -a -E "SECONDS|KEYWORD_SCORE"
  echo "--- video probe (8 frames) ---"
  python3 -u /tmp/a3_video_client.py "$PORT" qwen38 8 2>&1 | tail -5
  echo "VIDEO_RC=$?"
  echo "--- video probe (16 frames) ---"
  python3 -u /tmp/a3_video_client.py "$PORT" qwen38 16 2>&1 | grep -a -E "SECONDS|VIDEO_KEYWORD_SCORE|RESPONSE" | head -4
fi
stop_serve
echo C3_VIDEO_VERIFY_DONE
