#!/usr/bin/env bash
# gfx906-selfcheck -- verify the gfx906-critical pieces of this image.
#
#   gfx906-selfcheck            static + GPU checks
#   gfx906-selfcheck --no-gpu   skip the parts that touch the GPU
#
# Exit code 0 = everything passed. Each FAIL line names the artifact to fix.
set -uo pipefail

SP=/usr/local/lib/python3.12/dist-packages
fail=0
pass() { printf '  [ok]   %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }
warn() { printf '  [warn] %s\n' "$1"; }
chk()  { local d="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$d"; else bad "$d"; fi; }

if [ "${1:-}" = --no-gpu ]; then
  GPU=0
else
  GPU=1
fi

echo "== gfx906 image self-check"
echo "-- host visibility"
if ls /dev/dri/renderD* >/dev/null 2>&1 && [ -e /dev/kfd ]; then
  pass "$(ls -d /dev/dri/renderD* | wc -l) GPU render node(s) + /dev/kfd visible"
elif [ "$GPU" = 0 ]; then
  warn "no GPU devices (--no-gpu: assumed intentional, e.g. checking a build host)"
else
  bad "GPU devices not visible (--device=/dev/kfd --device=/dev/dri --group-add video)"
fi

echo "-- gfx906 runtime artifacts"
chk "gfx906 RCCL build (torch/lib/librccl.so.1.0)" test -f "$SP/torch/lib/librccl.so.1.0"
chk "RCCL symlink chain librccl.so -> librccl.so.1" test -L "$SP/torch/lib/librccl.so"
NTENS=$(ls "$SP/torch/lib/rocblas/library/" 2>/dev/null | grep -c gfx906)
if [ "${NTENS:-0}" -gt 100 ]; then pass "Tensile gfx906 kernels: $NTENS files"
else bad "Tensile gfx906 supplement files missing (found ${NTENS:-0}, need >100)"; fi
chk "W4A16 ExLlama bridge registered" grep -q WNA16ExllamaKernel "$SP/vllm/model_executor/kernels/linear/__init__.py"
chk "capture-safe gptq_gemm_out (python side)" grep -q gptq_gemm_out "$SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py"
chk "capture-safe gptq_gemm_out (compiled op)" grep -aq gptq_gemm_out "$SP/vllm/_C_stable_libtorch.abi3.so"
chk "Triton decode attention gfx906 BLOCK=32" grep -q on_gfx906 "$SP/vllm/v1/attention/ops/triton_decode_attention.py"
chk "Triton decode attention: no dangling is_gfx906 call" sh -c "! grep -q is_gfx906 '$SP/vllm/v1/attention/ops/triton_decode_attention.py'"
chk "gfx906 M=1 dense GEMV on skinny LLGemm1" grep -q use_gfx906_gemv "$SP/vllm/model_executor/layers/utils.py"
chk "MoE m==1 scalar-gate fast path" grep -q "acc = (x_view \* weight).sum" "$SP/vllm/model_executor/layers/utils.py"
chk "MoE tuned kernel config for this device" test -f "$SP/vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json"
chk "p2 GEMV .so present" test -s "$SP/gfx906/libp2gemv.so"
chk "p2 GEMV symbol exported" sh -c "nm -D --defined-only '$SP/gfx906/libp2gemv.so' | grep -qw p2_gemv"
chk "p2 dispatch wired" grep -q _P2_SHAPES "$SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py"
chk "rocBLAS GEMV fix module" test -f "$SP/gfx906_gemv_fix.py"
chk "GEMV fix auto-activation (.pth)" test -f "$SP/gfx906_gemv_fix.pth"

# Regression guard for the bug fixed on 2026-09-10: a patched file calling
# current_platform.<attr> where <attr> does not exist. Platform's base class has
# a __getattr__ fallback that logs a warning and returns None, so `hasattr()`
# is ALWAYS True and the call site dies much later with
#   TypeError: 'NoneType' object is not callable
# Only a dir()/__dict__-level existence check catches it. (Verified on this
# image: is_gfx906 -> in_dir=False but hasattr=True.)
python3 - <<'PY'
import pathlib, re, sys

SP = "/usr/local/lib/python3.12/dist-packages"
FILES = [
    "vllm/v1/attention/ops/triton_decode_attention.py",
    "vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py",
    "vllm/model_executor/kernels/linear/__init__.py",
    "vllm/model_executor/layers/utils.py",
]
refs = {}
for rel in FILES:
    p = pathlib.Path(SP) / rel
    if p.is_file():
        refs[rel] = sorted(set(
            re.findall(r"current_platform\.([A-Za-z_][A-Za-z0-9_]*)",
                       p.read_text(errors="ignore"))))
if not refs:
    print("  [FAIL] none of the patched python files exist")
    sys.exit(1)
try:
    from vllm.platforms import current_platform as cp
except Exception as e:  # noqa: BLE001
    print("  [warn] cannot import vllm.platforms (%s) -- guard skipped"
          % type(e).__name__)
    sys.exit(0)
known = set(dir(cp)) | set(vars(type(cp)))
dangling = [(r, a) for r, attrs in refs.items() for a in attrs if a not in known]
if dangling:
    for r, a in dangling:
        print("  [FAIL] %s calls current_platform.%s, which does not exist" % (r, a))
    sys.exit(1)
print("  [ok]   %d current_platform.<attr> reference(s) all resolve"
      % sum(len(v) for v in refs.values()))
PY
[ $? -eq 0 ] || fail=1

echo "-- versions"
python3 -c 'from importlib.metadata import version as v; print("  torch %-18s vllm %-14s triton %s" % (v("torch"), v("vllm"), v("triton")))' \
  && pass "python env importable" || bad "python env broken (importlib.metadata failed)"

if [ "$GPU" = 1 ]; then
  echo "-- gfx906 runtime env (image ENV; gfx906-serve backstops these too)"
  for kv in "HSA_OVERRIDE_GFX_VERSION=9.0.6" "VLLM_TRITON_ATTN_NUM_SEGMENTS=64" "VLLM_WNA16_EXLLAMA=1"; do
    k="${kv%%=*}"
    want="${kv#*=}"
    have="${!k:-}"
    if [ "$have" = "$want" ]; then
      pass "$k=$want"
    else
      warn "$k is '${have:-unset}' (want $want): raw vllm/python runs may be slow or broken; gfx906-serve forces $want"
    fi
  done

  echo "-- GPU behaviour (torch)"
  python3 - <<'PY'
import sys
import torch

ok = lambda m: print("  [ok]   %s" % m)
def bad(m):
    print("  [FAIL] %s" % m)
    sys.exit(1)

if not torch.cuda.is_available():
    bad("torch.cuda unavailable (driver / /dev/kfd problem)")
n = torch.cuda.device_count()
archs = [torch.cuda.get_device_properties(i).gcnArchName for i in range(n)]
if not all("gfx906" in a for a in archs):
    bad("visible devices are not gfx906: %s" % archs)
ok("%d gfx906 device(s): %s, hip %s" % (n, archs[0], torch.version.hip))

import gfx906_gemv_fix as g
if not g.APPLIED:
    bad("GEMV fix not applied (check .pth activation and VLLM_GFX906_NO_FIX)")
else:
    ok("GEMV fix applied (aten::mv/addmv -> GEMM path)")

# fp32 matrix*vector used to segfault the whole process on gfx906.
m = torch.randn(512, 512, device="cuda")
v = torch.randn(512, device="cuda")
r = torch.mv(m, v)
ref = m.double() @ v.double()
rel = (r.double() - ref).norm() / ref.norm()
if rel > 1e-3:
    bad("fp32 mv numerically wrong (rel=%.2e)" % rel.item())
else:
    ok("fp32 mv correct (rel=%.1e)" % rel.item())

a16 = torch.randn(256, 256, device="cuda", dtype=torch.float16)
if not torch.isfinite(a16 @ a16).all():
    bad("fp16 matmul produced non-finite values")
else:
    ok("fp16 matmul finite")

from vllm import _custom_ops as ops
if not hasattr(ops, "gptq_gemm_out"):
    bad("ops.gptq_gemm_out missing -- W4A16 CUDA-graph path unavailable")
else:
    ok("W4A16 graph-capture op exposed")
PY
  [ $? -eq 0 ] || fail=1
else
  echo "-- GPU behaviour skipped (--no-gpu)"
fi

echo
if [ "$fail" = 0 ]; then
  echo "SELFCHECK PASSED"
else
  echo "SELFCHECK FAILED -- fix the [FAIL] lines above before serving."
fi
exit $fail
