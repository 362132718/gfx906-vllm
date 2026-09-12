#!/usr/bin/env bash
# move_hook.sh -- put the census hook where the run actually happens, and take
# it back out of the delivery-candidate container.
set -uo pipefail
SP=/usr/local/lib/python3.12/dist-packages
SRC=/home/x/vllm-work/moe_tuning/lin_census_hook.py
RUN_CTR=${1:-p5-moe}

echo "=== remove from delivery candidate a2b-t1 ==="
docker exec a2b-t1 bash -lc "rm -f $SP/lin_census_hook.py $SP/lin_census_hook.pth; \
  ls $SP/lin_census_hook.* 2>/dev/null || echo '  removed (clean)'"

echo
echo "=== install into run container: $RUN_CTR ==="
docker cp "$SRC" "$RUN_CTR":/tmp/hook.py >/dev/null
docker exec "$RUN_CTR" bash -lc "
  SP=$SP
  cp /tmp/hook.py \$SP/lin_census_hook.py
  echo 'import lin_census_hook' > \$SP/lin_census_hook.pth
  python3 -c 'import torch.nn.functional as F; print(\"  no-op check:\", F.linear.__name__)'
  LIN_CENSUS_OUT=/tmp/x python3 -c 'import torch.nn.functional as F; print(\"  armed check:\", F.linear.__name__)'
"
echo "MOVE_HOOK_DONE"
