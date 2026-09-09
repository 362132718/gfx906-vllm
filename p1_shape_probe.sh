#!/usr/bin/env bash
# p1_shape_probe.sh -- ONE real serve with the shape-debug channel on, so the
# microbenchmark runs on shapes we observed rather than shapes we derived.
# Run inside the dev container. TP2 stays on because that is how the model
# actually serves; probe cost is one engine start.
#
# The dump does NOT happen by itself: wna16_exllama.py:70-85 registers
# _dump_shapes on atexit AND SIGUSR1, with the comment that atexit alone is not
# enough because vLLM hard-kills the TP workers on shutdown. So we signal the
# workers ourselves. The bracket in the pattern stops pkill from matching this
# very command line (memory gfx906-env-pitfalls #10).
set -uo pipefail
MODEL=${MODEL:-/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16}
OUTD=${OUTD:-/work/p1}
mkdir -p "$OUTD"
rm -f "$OUTD"/p1_shapes.csv*
export VLLM_WNA16_EXL_SHAPEDBG="$OUTD/p1_shapes.csv"
export VLLM_WNA16_EXLLAMA=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0        # engine core in-process; atexit reliable
# Payload MUST be a real file: spawn workers re-import the main module, and a
# stdin script has main module "<stdin>" => every worker dies with
# FileNotFoundError '/<stdin>' (reproduced on the first run of this probe).
python3 /work/p1/p1_shape_probe_main.py "$MODEL"
echo "shapedbg files (0-byte or missing = the dump never fired):"
ls -l "$OUTD"/p1_shapes.csv* 2>/dev/null || echo "NO SHAPE FILES"
echo P1_SHAPE_PROBE_DONE
