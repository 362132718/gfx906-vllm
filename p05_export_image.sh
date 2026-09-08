#!/usr/bin/env bash
# p05_export_image.sh -- run ON THE HOST, detached.
#
# Two jobs:
#  (1) Measure what docker save actually emits. The previous artifact was
#      compared against `docker image inspect .Size` (22.07 GB) and declared
#      "only 0.5% off", but `docker images` reports 69.8 GB for the same image.
#      One of those is not the size of the saved stream, so the compression
#      claim is unproven -- measure the raw byte count instead of inferring it.
#  (2) Export the CURRENT :latest (it carries the re-measured QUICKSTART) as the
#      rollback artifact, verify the gzip stream and its layer count by reading
#      it back through tar, and record a sha256.
set -uo pipefail
IMAGE=vllm-gfx906:latest
EXPORT_DIR=/home/x/image-export
ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
SHORT=${ID#sha256:}; SHORT=${SHORT:0:12}
TAR="$EXPORT_DIR/vllm-gfx906-$SHORT.tar.gz"

boot0=$(uptime -s)
echo "=== EXPORT START $(date -Is) image=$IMAGE id=$SHORT boot_since=$boot0"
mkdir -p "$EXPORT_DIR"
df -h "$EXPORT_DIR" | tail -1
echo "inspect_size_bytes=$(docker image inspect "$IMAGE" --format '{{.Size}}') layers=$(docker image inspect "$IMAGE" --format '{{len .RootFS.Layers}}')"
echo "docker_images_size=$(docker images --format '{{.Size}}' "$IMAGE")"

echo "--- (1) raw docker save stream length"
RAW=$(docker save "$IMAGE" | wc -c)
echo "raw_tar_bytes=$RAW"

echo "--- (2) save | pigz -> $TAR"
T0=$(date +%s)
docker save "$IMAGE" | pigz -p 6 > "$TAR" || { echo "FATAL: save failed"; exit 1; }
echo "save_pigz_seconds=$(( $(date +%s) - T0 ))"
BYTES=$(stat -c %s "$TAR")
echo "artifact_bytes=$BYTES"
awk -v r="$RAW" -v a="$BYTES" 'BEGIN{printf "compression: raw %.2f GB -> gz %.2f GB, saved %.1f%%, ratio %.2fx\n", r/1073741824, a/1073741824, 100*(1-a/r), r/a}'

echo "--- read-back: gzip integrity + layer count through tar (a rollback point"
echo "    that only loads because the layers are already in this store proves nothing)"
LB=$(date +%s)
LIST=$(pigz -dc "$TAR" | tar -tf -)
echo "read_back_seconds=$(( $(date +%s) - LB ))"
echo "$LIST" | grep -c 'layer\.tar$' | sed 's/^/layer_tar_members=/'
echo "$LIST" | wc -l | sed 's/^/tar_members=/'
echo "$LIST" | grep -c 'json$' | sed 's/^/json_manifest_members=/'
sha256sum "$TAR" | tee "$TAR.sha256"

echo "--- docker load round trip"
LOADED=$(docker load -i "$TAR" | tail -1)
echo "load_output=$LOADED"
[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$ID" ] && echo "LOADBACK_ID_MATCH $ID" || echo "!! LOADBACK_ID_MISMATCH"

echo "--- existing artifacts in $EXPORT_DIR"
ls -l "$EXPORT_DIR" | awk 'NR>1{printf "%s %.2f GB  %s\n", $5, $5/1073741824, $9}'
df -h "$EXPORT_DIR" | tail -1
echo "=== EXPORT DONE $(date -Is) boot_since=$(uptime -s)"
[ "$boot0" = "$(uptime -s)" ] && echo NO_REBOOT_DURING_RUN || echo "!! HOST_REBOOTED_MID_RUN"
echo P05_EXPORT_ALL_DONE
