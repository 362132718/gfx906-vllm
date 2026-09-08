#!/usr/bin/env bash
# kill_serve.sh -- stop any running C1 test + vllm server without matching our
# own command line (pkill -f patterns live in this file, not in argv).
pkill -f 'c1_serve_test\.sh' 2>/dev/null
pkill -f 'vllm serve' 2>/dev/null
pkill -f 'EngineCore' 2>/dev/null
# Workers re-exec with a renamed process title and get reparented to PID 1,
# so they survive the engine's death and keep holding VRAM.
pkill -f 'VLLM::Wor' 2>/dev/null
sleep 12
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM::Wor' 2>/dev/null
pkill -9 -f 'from multiprocessing' 2>/dev/null
sleep 4
n=$(ps -eo cmd | grep -c -E 'VLLM::[W]ork|vllm[ ]serv' || true)
echo "remaining_suspects=$n"
rocm-smi --showmeminfo vram | grep -i 'Used'
