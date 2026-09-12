#!/usr/bin/env bash
# install_hook.sh -- install the .pth-activated linear census hook into the
# container's site-packages so every spawned worker process loads it.
set -euo pipefail
SP=/usr/local/lib/python3.12/dist-packages
cp /tmp/hook.py "$SP/lin_census_hook.py"
echo 'import lin_census_hook' > "$SP/lin_census_hook.pth"
ls -la "$SP/lin_census_hook.py" "$SP/lin_census_hook.pth"
echo "--- dry run WITHOUT LIN_CENSUS_OUT (must be a no-op) ---"
python3 -c 'import torch.nn.functional as F; print("  name =", F.linear.__name__)'
echo "--- dry run WITH LIN_CENSUS_OUT (must be patched) ---"
LIN_CENSUS_OUT=/tmp/x python3 -c 'import torch.nn.functional as F; print("  name =", F.linear.__name__)'
echo "HOOK_INSTALL_OK"
