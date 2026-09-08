#!/usr/bin/env bash
# c3_matrix.sh -- bring up gfx906-serve with image=4 (matrix needs multi-image)
# and run the acceptance matrix. Delivery defaults otherwise (graphs ON).
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8171
export SERVED_NAME=qwen38
export MM='{"image":4,"video":1}'
LOG=/tmp/matrix_srv.log

gfx906-serve "$MODEL" > "$LOG" 2>&1 &
S=$!
rc=1
for i in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { rc=0; break; }
  kill -0 $S 2>/dev/null || { rc=2; break; }
  sleep 10
done
echo "MATRIX_SERVER_READY=${i}0s rc=$rc"
grep -a -oE "\[serve\].*" "$LOG" | head -1
grep -a -oE "GPU KV cache size: [0-9,]+ tokens.*" "$LOG" | tail -1
if [ $rc = 0 ]; then
  python3 -u /tmp/a3_matrix_client.py "$PORT" qwen38
  echo "MATRIX_RC=$?"
  grep -a -oE "Tried to allocate [0-9.]+ [KMGT]iB|OutOfMemoryError" "$LOG" | tail -2
fi
kill $S 2>/dev/null; sleep 8; kill -9 $S 2>/dev/null
bash /work/kill_serve.sh >/dev/null 2>&1
echo C3_MATRIX_DONE
