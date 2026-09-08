#!/usr/bin/env bash
# p0_sync.sh -- push local authoritative copies to the GPU host and verify by md5.
#
# The host dir /home/x/vllm-work is bind-mounted into the dev container as /work,
# so a file synced here is immediately importable inside the container at
# /work/p0/<name> -- no second docker cp step, no chance of two divergent copies.
#
# Usage: ./p0_sync.sh <relative-path-under-repo> [more...]
set -euo pipefail
HOST=${HOST:-x@192.168.10.11}
DIR=${P0_HOST_DIR:-/home/x/vllm-work/p0}
CONTAINER_DIR=${P0_CONTAINER_DIR:-/work/p0}

ssh -o BatchMode=yes "$HOST" "mkdir -p $DIR"
for f in "$@"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
  base=$(basename "$f")
  scp -q -o BatchMode=yes "$f" "$HOST:$DIR/$base"
  L=$(md5sum "$f" | cut -c1-32)
  R=$(ssh -o BatchMode=yes "$HOST" "md5sum $DIR/$base | cut -c1-32")
  if [ "$L" != "$R" ]; then echo "MISMATCH host copy $base local=$L host=$R" >&2; exit 1; fi
  # The mount is the same filesystem, but confirm the container actually sees it.
  C=$(ssh -o BatchMode=yes "$HOST" "docker exec vllm-gfx906 md5sum $CONTAINER_DIR/$base | cut -c1-32")
  if [ "$L" != "$C" ]; then echo "MISMATCH container view $base local=$L container=$C" >&2; exit 1; fi
  echo "ok   $base  $L  -> $CONTAINER_DIR/$base"
done
