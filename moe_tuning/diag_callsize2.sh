#!/usr/bin/env bash
echo "=== torch/_dynamo/utils.py: call_size（5543 附近）==="
awk 'NR>=5530 && NR<=5556 {printf "%d: %s\n", NR, $0}' /usr/local/lib/python3.12/dist-packages/torch/_dynamo/utils.py
echo
echo "=== torch/_dynamo/eval_frame.py:1435-1460 ==="
awk 'NR>=1435 && NR<=1460 {printf "%d: %s\n", NR, $0}' /usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py
echo
echo "=== torch/_compile.py: 45-58 ==="
awk 'NR>=45 && NR<=58 {printf "%d: %s\n", NR, $0}' /usr/local/lib/python3.12/dist-packages/torch/_compile.py
