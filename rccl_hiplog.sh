#!/bin/bash
export AMD_LOG_LEVEL=4
export NCCL_DEBUG=WARN
rm -f /tmp/h0.log /tmp/h1.log
env RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29671 python3 /tmp/rccl_probe.py > /tmp/h0.log 2>&1 &
p0=$!
sleep 2
env RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29671 python3 /tmp/rccl_probe.py > /tmp/h1.log 2>&1 &
p1=$!
wait $p0; wait $p1
echo "=== hip API calls that FAILED (rank0) ==="
grep -n -i "error\|Invalid" /tmp/h0.log | grep -v "^.*NCCL WARN" | head -30
echo "=== hip calls mentioning Module / Function / ext (rank0, first 25) ==="
grep -n -i "Module\|GetFunction\|extLaunch\|Kernel" /tmp/h0.log | head -25 | cut -c1-160
echo "=== tail rank0 ==="
tail -12 /tmp/h0.log | cut -c1-160
