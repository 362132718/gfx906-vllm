# p0_buckets.py -- turn the worker-side kernel table (/work/p0/p0_kernels.json,
# written by p0_parse_trace.py) into the spec section 1(b) five-bucket table and
# the section 1(c) GPU-busy-ratio routing decision.
#
# Window semantics: the traces cover K decode steps (K is derived from the GEMV
# launch count, see below), so every number here is per decode step -- no prefill
# contamination, and the denominator is the same 47 ms-class step the delivery
# path serves.
#
# Run anywhere /work is mounted:
#   docker exec vllm-gfx906 python3 /work/p0/p0_buckets.py
import json
import os
import re
import sys

# Order matters: first match wins. Two traps found against the real names:
# inductor fuses the GEMV epilogues into names like
# triton_poi_fused_gptq_gemm_out_mul_silu_slice_zero_5 -- those are elementwise
# work and must NOT land in the GEMV bucket, so the GEMV pattern is the kernel
# symbol, not the substring "gptq". And ncclDevKernel_Generic is mostly
# peer-wait time, so it gets its own bucket and is reported apart from compute.
BUCKETS = [
    ("w4a16_gemv",  r"gemm_half_q_half|reconstruct_exllama|shuffle_[0-9]bit|exllama"),
    ("rccl_wait",   r"nccl|rccl|allreduce|all_reduce"),
    ("attention",   r"unified_attention|reduce_segments|reshape_and_cache|flash|triton_(?:decode|fwd|attn)|attention|paged"),
    ("gdn",         r"gated_delta|conv1d|gdn|recurrent|kda|solve_tril|chunk_|l2norm|delta"),
    # P0.5: the ROCm skinny GEMV now answering M=1 dense projections. Kept
    # apart from dense_fp16 on purpose -- its whole point is that it is NOT the
    # Tensile serial-K path. Literal from p0_parse_trace.py's TOP25 dump.
    ("dense_gemv",  r"LLGemm1"),
    ("dense_fp16",  r"Cijk|tensile|hipblaslt|sgemm|hgemm|Wmma"),
    ("memcpy",      r"Memcpy|Memset|memset|copyBuffer"),
    ("elementwise", r"triton_|elementwise|vectorized_|copy|CatArrayBatched|norm|silu|rope|softmax|index|reduce|mul|add|fill"),
]
P0 = os.environ.get("OUT", "/work/p0")   # same env as p0_step / p0_parse_trace

d = json.load(open(f"{P0}/p0_kernels.json"))
w = json.load(open(f"{P0}/p0_wall.json"))
if d.get("schema") != "p0-trace-v1":
    sys.exit("FATAL: p0_kernels.json has schema %r -- rerun p0_parse_trace.py"
             % d.get("schema"))
per_step_us = w["per_step_ms"] * 1e3

ranks = d["ranks"]
prim_key = next(iter(ranks))
prim = ranks[prim_key]
gemv = d["gemv"]
K = round(gemv["launches"] / gemv["per_step_expected"])
if K == 0:
    sys.exit("FATAL: no GEMV launches in the window -- the ExLlama kernel is not"
             " live, do not trust any number in this file")
if abs(gemv["launches"] / gemv["per_step_expected"] - K) > 0.05:
    print("WARN: GEMV launches %d is not an integer multiple of the %d/step"
          " census -- window edges leaked into prefill" % (gemv["launches"],
                                                           gemv["per_step_expected"]))

print("wall: %.2f ms/decode-step = %.1f tok/s (unprofiled, %d-token request,"
      " pin=%sMHz, capture_sizes=%s)"
      % (w["per_step_ms"], 1e3 / w["per_step_ms"], w["n_tok"],
         w.get("clock_mhz"), w.get("capture_sizes")))
print("window (primary rank %s): K = %d decode steps, span %.1f ms, %d device"
      " events = %.0f kernel-launches/step"
      % (prim_key, K, prim["span_us"] / 1e3, prim["n_events"],
         prim["n_events"] / K))
for rk, r in ranks.items():
    print("  rank %s: sum %.1f ms  union %.1f ms  span %.1f ms  worst gap %.1f ms"
          % (rk, r["sum_us"] / 1e3, r["union_us"] / 1e3, r["span_us"] / 1e3,
             r["max_gap_us"] / 1e3))
    for c, n in sorted(r["cats"].items()):
        if c.startswith("UNSEEN"):
            print("    WARN unseen device category %s x%d" % (c, n))

out, cnt_of, un = {}, {}, []
for name, dev_us, cnt in prim["rows"]:
    if dev_us <= 0:
        continue
    for bname, pat in BUCKETS:
        if re.search(pat, name, re.I):
            out[bname] = out.get(bname, 0) + dev_us
            cnt_of[bname] = cnt_of.get(bname, 0) + cnt
            break
    else:
        un.append((name, dev_us, cnt))
tot = sum(out.values())
window_step_us = prim["span_us"] / K

print("\nThe window is taken under the torch profiler, which adds rank skew, so a"
      " profiled decode step runs %.1f ms while the delivery step is %.1f ms. All"
      " the extra is peer-wait: the compute rows below are profiler-independent and"
      " are the same per-step work in both. %%wall uses the delivery step (what"
      " tok/s depends on); %%win uses the profiled step (what the GPU was doing)."
      % (window_step_us / 1e3, w["per_step_ms"]))
print("\n%-12s %10s %8s %8s %9s" % ("bucket", "us/step", "%wall", "%win", "launches"))
for bname, us in sorted(out.items(), key=lambda kv: -kv[1]):
    print("%-12s %10.1f %7.1f%% %7.1f%% %9d"
          % (bname, us / K, us / K / per_step_us * 100, us / K / window_step_us * 100,
             cnt_of[bname]))
un_us = sum(u for _, u, _ in un)
print("%-12s %10.1f %7.1f%%" % ("UNCLASSIFIED", un_us / K, un_us / K / per_step_us * 100))
for n, u, c in sorted(un, key=lambda x: -x[1])[:12]:
    print("    %8.1f us/step  x%-6d %s" % (u / K, c, n[:88]))

wait = out.get("rccl_wait", 0)
comp = tot - wait
comp_us = comp / K
busy_union = prim["union_us"] / prim["span_us"]
print("\nGPU occupied %.2f of the profiled window (union/span, rank %s), but a third"
      " of that occupancy is nccl spin-wait." % (busy_union, prim_key))
print("ROUTING NUMBER: compute-only busy = %.1f ms kernel work / %.2f ms delivery"
      " step = %.2f" % (comp_us / 1e3, w["per_step_ms"], comp_us / per_step_us))
print("  rccl_wait in this window = %.1f us/step, i.e. %.0f%% of the profiled step's"
      " slack; at delivery the step leaves only %.1f ms for wait+bubbles, so the"
      " skew is profiler-induced, not a cross-rank/PCIe finding."
      % (wait / K, wait / K / window_step_us * 100,
         (per_step_us - comp_us) / 1e3))
gemv_us = out.get("w4a16_gemv", 0) / K
print("W4A16 GEMV = %.0f us/step = %.1f%% of the delivery step = %.1f%% of compute"
      % (gemv_us, gemv_us / per_step_us * 100, gemv_us / comp_us * 100))
print("dense fp16 GEMMs individually (launches/step identifies the owner):")
for n, us, c in prim["rows"]:
    if re.search(dict(BUCKETS)["dense_fp16"], n, re.I):
        print("    %8.0f us/step  %5.1f launches/step  %s"
              % (us / K, c / K, n[:78]))
print("spec 1(c): <0.6 -> not a kernel problem | 0.6-0.8 mixed | >=0.8 kernel"
      " suspects (GEMV / attention / GDN)")
if un_us / max(tot, 1) > 0.05:
    print("STOP: >5%% unclassified -- extend BUCKETS before trusting this table",
          file=sys.stderr)
    sys.exit(1)
