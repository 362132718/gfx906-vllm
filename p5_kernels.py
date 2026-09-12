"""p5_kernels.py -- bucket a vLLM per-worker chrome trace for the MoE model.

  python3 p5_kernels.py <prof_dir> [wall_json]

Prints (a) category buckets and (b) the top device kernels, both normalised to
one decode step. Deliberately does NOT reuse p0_parse_trace.py / p0_buckets.py:
those hardcode the 27B's ExLlama taxonomy and a GEMV calls-per-step anchor,
neither of which holds for a MoE.

Step-count anchor: the fused WNA16 MoE kernel runs twice per layer per step
(w13 then w2); all 40 layers in this model carry `mlp.experts`, so 80 calls/step.
We auto-detect the most frequent MoE-ish kernel and divide; if that fails we fall
back to device-busy / measured step time from wall_json.

Read this in eager mode: graphs only remove CPU launch gaps, kernel times are the
same, and eager keeps every kernel individually visible to the profiler.
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

prof = sys.argv[1] if len(sys.argv) > 1 else "/work/p5prof/prof"
wall = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(prof), "p5_wall.json")

HOST_CATS = {"cpu_op", "cuda_runtime", "hip_runtime", "cuda_driver",
             "gpu_user_annotation", "user_annotation", "overhead", "trace",
             "python_function"}

# strict: the fused expert GEMM only (2 per layer). Used both as the step anchor
# and to locate the end of the prefill forward.
MOE_RE_STRICT = re.compile(r"fused_moe_kernel|moe_kernel", re.I)

# ordered: first match wins, so put the specific ones above the generic ones
CATS = [
    ("MoE triton (wna16)", r"fused_moe|moe_kernel|_moe\b|experts?"),
    ("MoE align/topk/sum", r"moe_align|topk|radix|sort|moe_sum|moe_sort"),
    ("attention", r"unified_attention|context_attention|paged_attention|flash_attn|attn_fwd|decode_attention"),
    ("GDN / linear-attn", r"causal_conv1d|gated_delta|chunk|fla|mamba|conv1d|recurrent|gdn"),
    ("rocBLAS / Tensile GEMM", r"Cijk|rocblas|hipblas|tensile"),
    ("ExLlama W4A16 (dense)", r"gemm_half_q_half|exllama|gptq|dequant"),
    ("RCCL / all-reduce", r"rccl|nccl|AllReduce|ReduceScatter|AllGather"),
    ("norm", r"rms_norm|layer_norm|layernorm|norm_kernel|_norm\b"),
    ("activation", r"silu|swiglu|act_and_mul|gelu"),
    ("rope", r"rotary|rope"),
    ("elementwise / cast", r"elementwise|vectorized|cast|copy_|fill|cat_|index"),
    ("memcpy / memset", r"Memcpy|Memset|memcpy|memset"),
    ("sampling", r"sampl|argmax|logprob|penalt"),
]


def categorize(name: str) -> str:
    for label, pat in CATS:
        if re.search(pat, name, re.I):
            return label
    return "other"


def load(fn):
    try:
        raw = (gzip.open(fn, "rt", errors="ignore").read()
               if fn.endswith(".gz") else open(fn, errors="ignore").read())
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print("  !! %s: %s" % (os.path.basename(fn), e))
        return None


tot = defaultdict(float)
cnt = defaultdict(int)
cat_tot = defaultdict(float)
per_rank = {}

files = sorted(glob.glob(os.path.join(prof, "**", "*.json*"), recursive=True))
if not files:
    print("no traces under %s" % prof)
    sys.exit(1)

# DECODE-ONLY CUTOFF. A trace window opened by p5_prof.py contains one modest
# prefill followed by N decode steps. A prefill forward also fires exactly
# MOE_PER_FWD MoE calls, so "moe_calls / MOE_PER_FWD" counts forwards -- but the
# prefill's device time dwarfs all the decode steps combined (1600 tokens vs
# 1600 single tokens of work). Bucketing the raw window therefore reports the
# PREFILL's composition, not decode's. Cut everything before the end of the
# MOE_PER_FWD-th MoE call (= end of the prefill forward).
MOE_K = int(os.environ.get("MOE_PER_FWD", "80"))
SKIP_PREFILL = os.environ.get("SKIP_PREFILL", "1") == "1"

for fn in files:
    data = load(fn)
    if data is None:
        continue
    evs = data.get("traceEvents", data if isinstance(data, list) else [])
    cutoff = 0.0
    if SKIP_PREFILL:
        moe_spans = []
        for e in evs:
            if e.get("ph") != "X":
                continue
            if MOE_RE_STRICT.search(e.get("name") or ""):
                ts = float(e.get("ts") or 0.0)
                moe_spans.append((ts, ts + float(e.get("dur") or 0.0)))
        moe_spans.sort()
        if len(moe_spans) > MOE_K:
            cutoff = moe_spans[MOE_K - 1][1]      # end of the prefill forward
    rank_loc = defaultdict(float)
    first_ts, last_ts = None, None
    for e in evs:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat in HOST_CATS:
            continue
        if not cat.startswith(("kernel", "gpu", "device")):
            continue
        ts = float(e.get("ts") or 0.0)
        if ts < cutoff:
            continue
        dur = float(e.get("dur") or 0.0)
        if first_ts is None or ts < first_ts:
            first_ts = ts
        if last_ts is None or ts + dur > last_ts:
            last_ts = ts + dur
        n = e.get("name") or "?"
        tot[n] += dur
        cnt[n] += 1
        rank_loc[n] += dur
        cat_tot[categorize(n)] += dur
    if rank_loc:
        top = sorted(rank_loc.items(), key=lambda kv: -kv[1])[:2]
        per_rank[os.path.basename(fn)] = top
        if first_ts is not None:
            print("  [%s] decode-only window: %.1f ms"
                  % (os.path.basename(fn)[:34], (last_ts - first_ts) / 1000.0))

step_ms = None
if os.path.exists(wall):
    try:
        step_ms = json.load(open(wall)).get("decode_step_ms")
    except Exception:  # noqa: BLE001
        pass

busy = sum(tot.values())
moe_calls = max([c for n, c in cnt.items()
                 if re.search(r"moe|expert", n, re.I)] or [0])
steps = moe_calls / 80.0 if moe_calls >= 80 else None
if steps is None and step_ms:
    steps = busy / (step_ms * 1000.0)

print("trace files: %d" % len(files))
print("device-busy total: %.1f ms over %d kernels" % (busy / 1000.0, sum(cnt.values())))
print("step anchor: moe_calls=%d -> steps~%.1f" % (moe_calls, steps or 0))
if step_ms:
    print("measured wall step: %.2f ms  (device-busy/step = %.1f%%)"
          % (step_ms, (busy / steps / 1000.0) / step_ms * 100 if steps else 0))
print()
print("== CATEGORY buckets ==")
hdr = "%-26s %10s %10s %7s" % ("category", "total_ms", "us/step", "%busy")
print(hdr)
print("-" * len(hdr))
for label, us in sorted(cat_tot.items(), key=lambda kv: -kv[1]):
    print("%-26s %10.1f %10.2f %6.1f%%"
          % (label, us / 1000.0, (us / steps) if steps else float("nan"), us / busy * 100))
print()
print("== TOP kernels ==")
hdr = "%-50s %10s %8s %10s %7s" % ("kernel", "total_ms", "calls", "us/step", "%busy")
print(hdr)
print("-" * len(hdr))
def short(n):
    n = re.sub(r"^void (at::)?", "", n)
    n = re.sub(r"<.*?>", "", n)
    return n[:50]
for name, us in sorted(tot.items(), key=lambda kv: -kv[1])[:30]:
    print("%-50s %10.1f %8d %10.2f %6.1f%%"
          % (short(name), us / 1000.0, cnt[name],
             (us / steps) if steps else float("nan"), us / busy * 100))
print()
print("== per-trace top-1 (ranks should be symmetric) ==")
for f, top in list(per_rank.items())[:6]:
    print("  %-32s %s" % (f[:32], short(top[0][0]) if top else "-"))
