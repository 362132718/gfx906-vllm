#!/usr/bin/env bash
# 只读：验证待保留镜像的可用性（不碰 GPU、不加载模型）
IMG=vllm-gfx906:latest
SP=/usr/local/lib/python3.12/dist-packages

echo "=== 1) 待保留镜像元数据 ==="
docker inspect "$IMG" --format 'Id={{.Id}}
Created={{.Created}}
Entrypoint={{json .Config.Entrypoint}}
Cmd={{json .Config.Cmd}}
Size={{.Size}}'

echo
echo "=== 2) 镜像里的启动入口 ==="
docker run --rm --entrypoint bash "$IMG" -lc '
  echo "--- which gfx906-serve ---"; which gfx906-serve || echo "(不在 PATH)"
  echo "--- /usr/local/bin ---"; ls -l /usr/local/bin/
  echo "--- /opt/gfx906 ---"; ls -lR /opt/gfx906/ 2>/dev/null | head -40
'

echo
echo "=== 3) P0.6 补丁是否在镜像里 ==="
docker run --rm --entrypoint bash "$IMG" -lc "
  md5sum $SP/vllm/platforms/rocm.py
  echo -n 'cdna-fix hits: '; grep -c 'not _ON_GFX906) and any(' $SP/vllm/platforms/rocm.py
  echo -n 'fp8-guard hits: '; grep -c 'ROCm FP8 posture' $SP/vllm/platforms/rocm.py
"

echo
echo "=== 4) vllm / torch 版本（纯 import，不触 GPU）==="
docker run --rm --entrypoint bash "$IMG" -lc 'python3 -c "import vllm,torch;print(\"vllm\",vllm.__version__);print(\"torch\",torch.__version__)"' 2>&1 | tail -6

echo
echo "=== 5) 探针脚本是否在镜像里（p0 系列）==="
docker run --rm --entrypoint bash "$IMG" -lc 'ls -l $SP/../ 2>/dev/null | head -3; find / -xdev -maxdepth 4 -name "p0_*.py" 2>/dev/null | head'

echo
echo "=== 6) GPU 显存（确认上面几次 docker run 没占卡）==="
rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'GPU\[[01]\]'

echo
echo "=== 7) /models 与 /logs 在容器里挂的是什么（bind mount 声明）==="
docker inspect vllm --format '{{range .Mounts}}{{.Type}}  {{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'

echo
echo "=== 8) 当前 docker 全量 ==="
docker images --format '{{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'
echo "--- 容器 ---"
docker ps -a --format '{{.Names}}  {{.Image}}  {{.Status}}'

echo
echo "=== 9) docker 空间明细 ==="
docker system df
