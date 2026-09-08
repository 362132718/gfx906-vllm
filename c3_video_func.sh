#!/usr/bin/env bash
# c3_video_func.sh -- functional video check on the UNCAPPED config (video:1,
# model's own size budget). Startup already proven OK (420s eager); the real
# question is whether an actual video request encodes without OOM and gets the
# right answer, and whether images still work in the same server.
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8162
export EAGER=1
export MM='{"image":1,"video":1}'
export SERVED_NAME=qwen38          # clients ask for "qwen38"
LOG=/tmp/vf_$1.log

gfx906-serve "$MODEL" > "$LOG" 2>&1 &
S=$!
rc=1
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then rc=0; break; fi
  kill -0 $S 2>/dev/null || { rc=2; break; }
  sleep 10
done
[ $rc = 0 ] && echo "UNCAPPED_RESULT=STARTED after ~$((i*10))s" || echo "UNCAPPED_RESULT=NOT_READY($rc)"

if [ $rc = 0 ]; then
  echo "--- image probe ---"
  python3 -u /tmp/a3_client.py "$PORT" 2>&1 | grep -aE "SECONDS|KEYWORD_SCORE|Error" | head -3
  for N in 4 8 16; do
    echo "--- video probe frames=$N ---"
    python3 -u /tmp/a3_video_client.py "$PORT" qwen38 $N 2>&1 | tail -6
    echo "VIDEO_RC=$?"
    grep -a -oE "Tried to allocate [0-9.]+ [KMGT]iB|OutOfMemoryError" "$LOG" | tail -2
  done
fi
kill $S 2>/dev/null; sleep 8; kill -9 $S 2>/dev/null
bash /work/kill_serve.sh >/dev/null 2>&1
echo C3_VIDEO_FUNC_DONE
