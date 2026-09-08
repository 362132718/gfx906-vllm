#!/bin/bash
# Try several RCCL transport configurations. Prints PASS/FAIL per case.
run_case() {
    local name="$1"
    shift
    rm -f /tmp/c0.log /tmp/c1.log
    env "$@" RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 \
        python3 /tmp/rccl_probe.py > /tmp/c0.log 2>&1 &
    local p0=$!
    sleep 2
    env "$@" RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 \
        python3 /tmp/rccl_probe.py > /tmp/c1.log 2>&1 &
    local p1=$!
    wait $p0; local e0=$?
    wait $p1; local e1=$?
    if grep -q "RCCL_TP2_TEST: PASS" /tmp/c0.log; then
        echo "CASE[$name]: PASS"
    else
        echo "CASE[$name]: FAIL e0=$e0 e1=$e1 :: $(grep -i -m1 'error\|Error' /tmp/c0.log | head -c 160)"
    fi
}

run_case "p2p_disable" NCCL_P2P_DISABLE=1
run_case "shm_disable" NCCL_SHM_DISABLE=1
run_case "p2p+shm_disable" NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
run_case "fine_grain" HSA_FORCE_FINE_GRAIN_PCIE=1
run_case "no_sdma" HSA_ENABLE_SDMA=0
run_case "1channel" NCCL_MAX_NCHANNELS=1 NCCL_MIN_NCHANNELS=1
run_case "simple_proto" NCCL_PROTO=Simple
run_case "tree_algo" NCCL_ALGO=Tree
