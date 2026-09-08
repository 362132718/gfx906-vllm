#!/usr/bin/env bash
# p05_ship_doc.sh -- run ON THE HOST. Propagate a docker/entrypoint/* edit all the
# way into the delivery image.
#
# Why this is not just "rebuild": prepare_build_context.sh step 4 does
# `docker cp $CONTAINER:/opt/gfx906/.`, i.e. the entrypoint flows container ->
# build context, so a repo-only edit never reaches Path B. And Path A copies
# entrypoint/ from the docker/ dir, so BOTH copies have to be updated.
# Usage: p05_ship_doc.sh <expected-md5>
set -uo pipefail
WANT="$1"
HOST_DIR=/home/x/vllm-work/docker
CONTAINER=vllm-gfx906
FILE=QUICKSTART.md
P=0

mfile() { docker exec "$CONTAINER" md5sum "/opt/gfx906/$FILE" | cut -c1-32; }

echo "=== SHIP DOC START $(date -Is) expect=$WANT"
[ -f "$HOST_DIR/entrypoint/$FILE" ] || { echo "FATAL: $HOST_DIR/entrypoint/$FILE missing (sync it first)"; exit 1; }
got=$(md5sum "$HOST_DIR/entrypoint/$FILE" | cut -c1-32)
echo "host-context: $got"; [ "$got" = "$WANT" ] || { echo "FATAL: host copy mismatch"; exit 1; }

docker cp "$HOST_DIR/entrypoint/$FILE" "$CONTAINER:/opt/gfx906/$FILE"
got=$(mfile); echo "container:    $got"; [ "$got" = "$WANT" ] || { echo "FATAL: container mismatch"; exit 1; }

cd "$HOST_DIR" || exit 1
bash prepare_build_context.sh "$CONTAINER" > /home/x/vllm-work/p0/p05_prepare_doc.log 2>&1
echo "prepare rc=$?"; tail -3 /home/x/vllm-work/p0/p05_prepare_doc.log
got=$(md5sum build-context/entrypoint/$FILE | cut -c1-32)
echo "build-context: $got"; [ "$got" = "$WANT" ] || { echo "FATAL: build-context mismatch"; exit 1; }

bash /home/x/vllm-work/p0/p05_build_both.sh
[ "$?" = 0 ] || { echo "FATAL: build failed"; exit 1; }

echo "--- md5 inside the delivered image"
docker run --rm vllm-gfx906:latest md5sum "/opt/gfx906/$FILE" | cut -c1-32 | sed 's/^/image:         /'
docker run --rm vllm-gfx906:latest md5sum "/opt/gfx906/$FILE" | grep -q "^$WANT" || P=1

echo "--- selfcheck of the rebuilt image on real GPUs"
docker run --rm --device /dev/kfd --device /dev/dri -v /home/x/models:/home/x/models:ro \
  vllm-gfx906:latest gfx906-selfcheck > /home/x/vllm-work/p0/p05_selfcheck_doc.log 2>&1
rc=$?
echo "SELFCHECK_RC=$rc ok=$(grep -c '\[ok\]' /home/x/vllm-work/p0/p05_selfcheck_doc.log) fail=$(grep -c FAIL /home/x/vllm-work/p0/p05_selfcheck_doc.log)"
[ "$rc" = 0 ] || P=1

echo "=== SHIP DOC DONE $(date -Is) P=$P $( [ "$P" = 0 ] && echo ALL-OK || echo PROBLEM ) "
