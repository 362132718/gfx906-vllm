#!/usr/bin/env bash
# p2_t6_serve.sh -- G-P6: delivery-clock (sclk auto, UNLOCKED) serve tok/s with
# the image a customer would run, two arms in ONE fresh container, one variable
# (VLLM_WNA16_P2_GEMV).
#
# Why unlock: the 32.4 tok/s baseline this gate is defined against was taken at
# delivery clocks, and a 1425-pinned number is not comparable to it (pitfall #31
# cuts both ways -- the G-P5 A/B pins clocks, this test must not).
#
# Why two arms here: the on-vs-off delta is also the only dispatch evidence this
# path produces. The serve log cannot show _P2_HITS (no export path, pitfall #26)
# and no device trace is taken, so if the two arms come out equal the P2 route
# did not survive the move from engine harness to serve flags -- report it, do
# not call that a pass.
#
# The cache is NOT isolated between arms: after the G-P3 fix the env read lives
# inside the custom-op impl, which dynamo does not trace, so both arms compile to
# the same artifact. Arm "on" runs first and warms the cache arm "off" then
# reuses; if the second boot does not get much faster, that reuse assumption is
# what to look at first.
#
# WHY THIS SCRIPT IS RESUMABLE (run 1 evidence): at 04:31 on 2026-09-10 the HOST
# lost power mid-compile -- previous-boot journal ends with no shutdown message,
# "user.journal corrupted or uncleanly shut down", and every container on the box
# including the unrelated dev one shows Exited (255) at the same second. Nothing
# in the P2 path caused it; it is [[gfx906-host-instability]]. A detached script
# does not survive that, so: completed arms are skipped, the container (and with
# it the writable layer holding the compile cache) is reused instead of recreated,
# and a failed boot retries in-place. Re-run the same command after a reboot.
set -uo pipefail
IMAGE=vllm-gfx906:latest
CNAME=p2-t6
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
WORK=/home/x/vllm-work
OUT=$WORK/p2t6
PORT=8000
MAX_TRIES=${MAX_TRIES:-3}
mkdir -p "$OUT"

echo "=== T6 START $(date -Is)"
cd "$WORK" || exit 1
# The helper lives in p0/, and an unlock that did not happen invalidates every
# number this test produces (a 1425-pinned step time is not comparable with the
# 32.4 taken at delivery clocks), so verify the state instead of trusting rc.
CLOCK=/home/x/vllm-work/p0/p0_clock.sh
"$CLOCK" unlock || { echo "FATAL: unlock command failed"; exit 1; }
echo "--- clocks now:"; st=$("$CLOCK" status) || { echo "FATAL: status failed"; exit 1; }
echo "$st"
n_auto=$(echo "$st" | grep -c "level=auto")
[ "$n_auto" = 2 ] || { echo "FATAL: $n_auto/2 GPUs at auto -- refusing to measure at pinned clocks"; exit 1; }

# Reuse, do not recreate: `docker rm -f` would throw away the compile cache that
# a partially-successful boot paid for, turning every retry back into a 700 s
# cold start. Recreate only when the container is absent or built from a
# different image id than the one under test.
want_img=$(docker inspect --format '{{.Id}}' "$IMAGE")
have=$(docker inspect --format '{{.Image}} {{.State.Status}}' "$CNAME" 2>/dev/null || echo " none")
have_img=${have%% *}
if [ "$have_img" != "$want_img" ]; then
  echo "--- ($CNAME image=$have_img != $IMAGE id=$want_img) recreating"
  docker rm -f "$CNAME" >/dev/null 2>&1
  docker run -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --shm-size 16g --ipc=host \
    -v /home/x/models:/home/x/models:ro \
    -v "$WORK":"$WORK" \
    "$IMAGE" sleep infinity >/dev/null || { echo "FATAL: docker run failed"; exit 1; }
else
  echo "--- reusing $CNAME (image matches)"
  docker start "$CNAME" >/dev/null 2>&1
fi
echo "--- container: $(docker inspect "$CNAME" --format '{{.Image}} {{.State.Status}}')"

for arm in on off; do
  if grep -q "DECODE256 best=" "$OUT/probe_$arm.txt" 2>/dev/null; then
    echo "=== arm=$arm SKIP (already measured): $(grep -E 'REQ2_WARM|DECODE256 best' "$OUT/probe_$arm.txt" | tr '\n' ' ')"
    continue
  fi
  # off gets the -e flag; on must NOT get one, so that arm "on" is literally the
  # out-of-the-box image (the impl's own default is "1").
  if [ "$arm" = off ]; then ENVX=(-e VLLM_WNA16_P2_GEMV=0); else ENVX=(); fi
  echo "=== arm=$arm p2=${ENVX[*]:-unset(=1)} $(date -Is)"
  docker exec -i "${ENVX[@]}" "$CNAME" bash -s -- "$MODEL" "$PORT" "$OUT" "$arm" "$MAX_TRIES" <<'INSIDE'
set -uo pipefail
MODEL=$1; PORT=$2; OUT=$3; ARM=$4; MAX_TRIES=$5
echo "env in this exec: VLLM_WNA16_P2_GEMV=${VLLM_WNA16_P2_GEMV:-unset(=> impl default 1)}"
PROBE=$OUT/probe_$ARM.txt

pgrep -f 'vllm serve|VLLM::Wor' | xargs -r kill -9 2>/dev/null
for i in $(seq 1 18); do
  free=$(python3 -c 'import torch;print(min(torch.cuda.mem_get_info(i)[0] for i in (0,1))/1e9)' 2>/dev/null || echo 0)
  awk -v f="$free" 'BEGIN{exit !(f>16.0)}' && { echo "preflight_vram_free_minGB=$free"; break; }
  echo "waiting for VRAM: ${free} GB"; sleep 10
done
echo "cache: $(du -sh /root/.cache/vllm* 2>/dev/null | awk '{printf "%s(%s) ",$1,$2}' || echo none)"

try=1
while [ "$try" -le "$MAX_TRIES" ]; do
  LOG=$OUT/serve_${ARM}_try$try.log
  echo "--- try $try/$MAX_TRIES log=$LOG $(date -Is)"
  gfx906-serve "$MODEL" > "$LOG" 2>&1 &
  S=$!
  T0=$(date +%s); READY=0
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
    kill -0 $S 2>/dev/null || { echo "TRY_${try}_DIED after $(( $(date +%s) - T0 ))s"; tail -30 "$LOG"; break; }
    sleep 10
  done
  if [ "$READY" = 1 ]; then break; fi
  kill -9 $S 2>/dev/null; pgrep -f 'VLLM::Wor' | xargs -r kill -9 2>/dev/null; sleep 60
  try=$(( try + 1 ))
done
if [ "$READY" != 1 ]; then
  echo "ARM_FAILED: $try-1 tries, last 40 lines:"; tail -40 "${LOG}"; exit 1
fi
echo "READY_AFTER=$(( $(date +%s) - T0 ))s try=$try"
grep -a -oE "torch.compile took [0-9.]+ s" "$LOG" | head -1
grep -a -oE "GPU KV cache size: [0-9,]+ tokens.*" "$LOG" | head -1
grep -a -c "Using WNA16ExllamaKernel" "$LOG" | sed 's/^/wna16_exllama_lines=/'
NAME=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
       | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "resolved_model_id=$NAME"

# Same probe, same prompts, same order as p05_serve_probe.sh: REQ2_WARM (64 tok)
# is the metric the 32.4 baseline was read from.
#
# Do NOT write this as `python3 - ... | tee "$PROBE" <<'PY'`: the heredoc binds to
# the LAST pipeline element, so tee received the source and python received EOF --
# run 1 produced a probe file full of python code and rc=120. Hence redirect to
# the file, then cat. The grep is the falsifiability half: an empty probe must not
# read as a completed arm.
python3 - "$PORT" "$NAME" <<'PY' > "$PROBE" 2>&1
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
def ask(mt, prompt):
    body = json.dumps({"model": name, "prompt": prompt,
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1200))
    return d, time.time() - t0
d, el = ask(24, "The capital of France is")
print("REQ1: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
d, el = ask(64, "The capital of France is")
print("REQ2_WARM: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
rates = []
for i in range(3):
    d, el = ask(256, "请用一句话解释为什么显卡的显存带宽会限制大模型推理速度。")
    r = d["usage"]["completion_tokens"] / el
    rates.append(r)
    print("DECODE256 rep%d: %.1f tok/s" % (i + 1, r))
print("DECODE256 best=%.1f mean=%.1f" % (max(rates), sum(rates) / len(rates)))
PY
prc=$?
grep -q "DECODE256 best=" "$PROBE" || { echo "PROBE_EMPTY rc=$prc:"; cat "$PROBE"; exit 1; }
cat "$PROBE"

kill $S 2>/dev/null; sleep 10; kill -9 $S 2>/dev/null
pgrep -f 'VLLM::Wor' | xargs -r kill -9 2>/dev/null
sleep 30
echo "remaining vllm procs: $(pgrep -cf 'vllm serve|VLLM::Wor' || echo 0)"
INSIDE
  rc=$?
  echo "=== arm=$arm rc=$rc $(date -Is)"
  [ "$rc" = 0 ] || { echo "T6 ABORTED (arm $arm) -- re-run this script to resume"; exit "$rc"; }
  sleep 300                        # the pair thermally trips; serve is heavier than the harness
done

echo "=== G-P6 GATE $(date -Is)"
python3 - "$OUT" <<'PY'
import re, sys
out = sys.argv[1]
GATE, BASE = 33.4, 32.4
def read(arm):
    path = "%s/probe_%s.txt" % (out, arm)
    t = open(path).read()
    def grab(pat):
        m = re.search(pat, t)
        assert m, "no %r in %s" % (pat, path)
        return float(m.group(1))
    return {"REQ2_WARM": grab(r"REQ2_WARM: ([0-9.]+)"),
            "DECODE256_mean": grab(r"DECODE256 best=[0-9.]+ mean=([0-9.]+)")}
on, off = read("on"), read("off")
d = on["REQ2_WARM"] - off["REQ2_WARM"]
print("arm on  REQ2_WARM=%.1f  DECODE256_mean=%.1f" % (on["REQ2_WARM"], on["DECODE256_mean"]))
print("arm off REQ2_WARM=%.1f  DECODE256_mean=%.1f  (= the new delivery baseline at auto clocks)" % (off["REQ2_WARM"], off["DECODE256_mean"]))
print("P2 delta = %+.1f tok/s (%.2f%%)  [G-P5 device-side predicted +0.79 ms/step ~ +2.4%%]" % (d, 100 * d / off["REQ2_WARM"]))
print("%s G-P6 on-arm vs gate: %.1f vs >= %.1f  (baseline before P2 was %.1f at the same probe)"
      % ("PASS" if on["REQ2_WARM"] >= GATE else "FAIL", on["REQ2_WARM"], GATE, BASE))
PY
echo "=== T6 DONE $(date -Is)"
