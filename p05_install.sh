#!/usr/bin/env bash
# p05_install.sh -- install one repo file over an absolute path INSIDE the dev
# container and prove local == host == container by md5.
#
# p0_sync.sh only covers dev scripts under /work/p0; a patched site-packages
# file needs a different target, and "the container still has the old copy" is
# a bug class that already cost a measurement cycle. Refuses to install anything
# the delivery recipe does not COPY, so a change can't silently live only in
# the dev container.
#
# Usage: ./p05_install.sh <repo-path> <container-abs-path>
set -euo pipefail
HOST=${HOST:-x@192.168.10.11}
CONTAINER=${CONTAINER:-vllm-gfx906}
SRC=${1:?repo path}; DST=${2:?container absolute path}
[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
# The recipe spells the same target two ways: ${SP}/<rel> in COPY lines, the
# absolute path in test/grep assertions. Accept either, but refuse a file the
# delivery recipe never mentions. Fixed-string matching only -- an ERE here
# needs escaping, and sed's replacement-side & expands to the whole match.
REL=${DST#/usr/local/lib/python3.12/dist-packages/}
if ! grep -qF -e "$DST" -e "dist-packages/${REL}" -e '${SP}/'"${REL}" \
       docker/Dockerfile.rocm_gfx906; then
  echo "FATAL: $DST is not COPYed by docker/Dockerfile.rocm_gfx906 -- this edit" \
       "would not enter the delivery image. Do Task 6 first, or FORCE=1." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
fi
NAME=$(basename "$SRC")
scp -q -o BatchMode=yes "$SRC" "$HOST:/home/x/vllm-work/p0/$NAME"
ssh -o BatchMode=yes "$HOST" "docker cp /home/x/vllm-work/p0/$NAME $CONTAINER:$DST"
L=$(md5sum "$SRC" | cut -c1-32)
C=$(ssh -o BatchMode=yes "$HOST" "docker exec $CONTAINER md5sum $DST | cut -c1-32")
[ "$L" = "$C" ] || { echo "MISMATCH $SRC local=$L container=$C" >&2; exit 1; }
echo "ok  $SRC -> $CONTAINER:$DST  $L"
