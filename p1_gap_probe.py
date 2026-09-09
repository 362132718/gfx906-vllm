# p1_gap_probe.py -- review-P0-4's decisive experiment, plus P0-3's incumbent fix.
#
#   [1] load-only probe vs qweight-only floor, per shape: if pure streaming with
#       the same access pattern also sits far above the floor, the floor itself
#       is missing a shape-dependent term and "x-of-floor" percentages mislead.
#   [2] incumbent with double-buffer rotation (same L2 discipline as the
#       prototypes) -- the old baseline was single-buffer, which systematically
#       flattered it on the small-weight shape (3072,5120).
#   [3] V3 with unroll {1,2} on the losing families (the sweep bottomed at u=4).
#   [4] VGPR/SGPR per kernel from the code object notes (llvm-readelf -n).
import json, os, re, subprocess, sys
os.environ["VLLM_WNA16_EXLLAMA"] = "1"
import torch
from torch.utils.cpp_extension import load

mod = load(name="p1_kernels", sources=["/work/p1/p1_kernels.cu"],
           extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
           build_directory="/tmp/p1_bench_build", verbose=False)
sys.path.insert(0, "/work/p1")
import p1_l1_bench as b

FLOORS = json.load(open("/work/p1/p1_floors.json"))
BW = FLOORS["bw_pure_read_gb_s"]
out = {"loadonly": {}, "incumbent_rot": {}, "v3_lowunroll": {}}

print("=== [1] load-only probe vs qweight-only floor ===", flush=True)
for key, meta in sorted(FLOORS["shapes"].items()):
    K, N = (int(v) for v in key.split(","))
    x, qw, sc, zr = b.tensors(K, N, 1)
    sink = torch.zeros(64, device=b.DEV, dtype=torch.int32)
    best = None
    for sk in (1, 2, 4, 8):
        if (K // 8) % sk:
            continue
        fn = lambda sk=sk: mod.p1_loadonly_launch(qw, sink, sk)
        t = b.timeit(b.graphed(fn), warm=8, reps=32, R=8)
        best = t if best is None else min(best, t)
    qw_floor = (K // 8) * N * 4 / (BW * 1e9) * 1e6
    out["loadonly"][key] = dict(us=best, qw_floor_us=qw_floor,
                                pct_floor=round(100 * qw_floor / best, 1))
    print("loadonly K=%-6d N=%-6d best=%.2f us vs qw-floor=%.2f us => %.1f%% of floor"
          % (K, N, best, qw_floor, 100 * qw_floor / best), flush=True)

print("=== [2] incumbent WITH rotation (the old baseline was single-buffer) ===", flush=True)
for key, meta in sorted(FLOORS["shapes"].items()):
    K, N = (int(v) for v in key.split(","))
    bufs = [b.tensors(K, N, 1), b.tensors(K, N, 2)]
    from vllm import _custom_ops as ops
    c_out = torch.zeros(1, N, device=b.DEV, dtype=torch.float16)
    temp_dq = torch.empty(0, dtype=torch.float16, device=b.DEV)
    g_idx = torch.empty((0,), dtype=torch.int32, device=b.DEV)
    state = [0]

    def once():
        q = bufs[state[0] % 2]
        state[0] += 1
        c_out.zero_()
        ops.gptq_gemm_out(c_out, q[0].view(1, K).contiguous(), q[1], q[3], q[2],
                          g_idx, True, True, 4, temp_dq)
        return c_out
    e = b.timeit(once)
    gr = b.timeit(b.graphed(once))
    old = json.load(open("/work/p1/p1_incumbent.json"))["%d,%d" % (K, N)]
    out["incumbent_rot"][key] = dict(eager_us=e, graph_us=gr,
                                     old_graph_us=old["graph_us"])
    print("incumbent-rot K=%-6d N=%-6d eager=%.2f graph=%.2f us (single-buffer was %.2f)"
          % (K, N, e, gr, old["graph_us"]), flush=True)

print("=== [3] V3 with unroll {1,2} on the losing families ===", flush=True)
for key in ("5120,17408", "8704,5120", "3072,5120"):
    K, N = (int(v) for v in key.split(","))
    x, qw, sc, zr = b.tensors(K, N, 1)
    floor_us = FLOORS["shapes"][key]["floor_us"]
    for sk in (2, 4, 8):
        if (K // b.G) % sk:
            continue
        for u in (1, 2, 4):
            scratch = torch.zeros(sk * N, device=b.DEV, dtype=torch.float32)
            fn = lambda: mod.p1_gemv(x, qw, sc, zr, scratch, 3, sk, u, True)
            try:
                if not b.check(3, sk, u, K, N):
                    continue
                gm = b.timeit(b.graphed(fn))
                out["v3_lowunroll"]["%s/sk%d/u%d" % (key, sk, u)] = gm
                print("V3 K=%-6d N=%-6d sk=%d u=%d graph=%.2f us (%.1f%% of floor)"
                      % (K, N, sk, u, gm, 100 * floor_us / gm), flush=True)
            except Exception as ex:
                print("  skip", key, sk, u, str(ex)[:80], flush=True)

print("=== [4] VGPR/SGPR per kernel (llvm-readelf notes) ===", flush=True)
notes = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "--notes",
                        "/tmp/p1_bench_build/p1_kernels.cuda.o"],
                       capture_output=True, text=True)
txt = notes.stdout or notes.stderr
for m in re.finditer(r"\.name:\s*'([^']+)'[\s\S]{0,400}?\.vgpr_count:\s*(\d+)"
                     r"[\s\S]{0,200}?\.sgpr_count:\s*(\d+)", txt):
    print("VGPR %-42s vgpr=%-4s sgpr=%s" % (m.group(1)[:42], m.group(2), m.group(3)))
if "vgpr" not in txt.lower():
    print("(no amdhsa vgpr metadata found; first 200 chars:) %s" % txt[:200])

json.dump(out, open("/work/p1/p1_gap_probe.json", "w"), indent=1)
print("GAP-PROBE-DONE")
