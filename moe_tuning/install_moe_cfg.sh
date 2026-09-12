#!/usr/bin/env bash
# install_moe_cfg.sh -- drop the tuned MoE kernel config into a container's
# vLLM configs dir. This is the delivery path too: prepare_build_context.sh
# tars the WHOLE dist-packages (line 87-88), so a file placed here rides the
# existing 18 GB tarball into the image with no Dockerfile change.
#
# Usage: bash install_moe_cfg.sh <container> [src.json]
set -uo pipefail

CTR=${1:?usage: install_moe_cfg.sh <container> [src.json]}
SRC=${2:-/home/x/vllm-work/moe_tuning/configs/E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json}
SP=/usr/local/lib/python3.12/dist-packages
DST="$SP/vllm/model_executor/layers/fused_moe/configs"

[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
docker cp "$SRC" "$CTR:/tmp/moecfg.json" >/dev/null || exit 1
docker exec "$CTR" bash -lc "mkdir -p '$DST' && cp /tmp/moecfg.json '$DST/' && ls -la '$DST/' | grep -i 'AMD_Radeon_Pro_VII'"
echo "md5 (host)      : $(md5sum "$SRC" | awk '{print $1}')"
docker exec "$CTR" bash -lc "md5sum '$DST/$(basename "$SRC")'"
echo "INSTALL_MOE_CFG_OK"
