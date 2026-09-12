# p2_7168_bistability.py -- the 7168 family has TWO timing states ~0.70 us apart
# (58.10 / 58.80) and a single process switches between them mid-run
# (logs from p2_gate_time_contamination.py: CLEAN r0-r2 = 58.10, r3-r4 = 58.81,
# both code paths moving together to within 0.01 us). 8192 does not do this.
#
# Why it matters: the user's approved G-P0 criterion is "cumulative benefit >= 99%
# of the 1.3x promise (1148.7 us)". Recomputed from the production gate numbers,
# the LOW mode gives 1145.4 us (99.7%) and the HIGH mode gives 1136.3 us (98.9%) --
# i.e. the pass/fail line sits INSIDE this 0.7 us state noise, so it must be
# characterised, not averaged away.
#
# Question this answers: does the switch track the actual clocks (freq*_input is the
# LIVE value, unlike pp_dpm_* which just echoes the forced DPM level), the temperature,
# or neither (=> page/L2 placement or a driver state we do not control)?
import statistics
import sys
import time

sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l1_bench as b
import p2_shim

SK, U = 10, 8
K, N = 5120, 7168
ROUNDS = 12


def live():
    """LIVE clocks (Hz -> MHz), edge temperature, busy %. pp_dpm_* would only
    repeat the lock request, so read the hwmon inputs."""
    import glob
    h = sorted(glob.glob("/sys/class/drm/card0/device/hwmon/hwmon*"))[0]
    r = {}
    for lab, key in (("freq1_input", "sclk"), ("freq2_input", "mclk"),
                     ("temp1_input", "temp_mC"), ("power1_input", "power_uW")):
        try:
            r[key] = int(open("%s/%s" % (h, lab)).read().strip())
        except Exception:
            r[key] = -1
    try:
        r["busy"] = int(open("/sys/class/drm/card0/device/gpu_busy_percent").read())
    except Exception:
        r["busy"] = -1
    return "sclk=%d mclk=%d edge=%d.%dC pwr=%.1fW busy=%d%%" % (
        r["sclk"] // 1000000, r["mclk"] // 1000000, r["temp_mC"] // 1000,
        (r["temp_mC"] // 100) % 10, r["power_uW"] / 1e6, r["busy"])


bufs = [b.tensors(K, N, seed=1), b.tensors(K, N, seed=2)]
scratch = torch.zeros(SK * N, device=b.DEV, dtype=torch.float32)
part = torch.zeros((SK, N), device=b.DEV, dtype=torch.float32)
out = torch.zeros(N, device=b.DEV, dtype=torch.float16)
graphs = {}
for name in ("prod", "proto"):
    state = [0]

    def fn():
        t = bufs[state[0] % 2]
        state[0] += 1
        if name == "prod":
            p2_shim.p2_gemv(t[0], t[1], t[2], t[3], part, out, K, N, SK)
        else:
            b.mod.p1_gemv(t[0], t[1], t[2], t[3], scratch, 3, SK, U, True)

    fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    gr.replay()
    torch.cuda.synchronize()
    graphs[name] = gr

vals = {"prod": [], "proto": []}
for r in range(ROUNDS):
    order = ["prod", "proto"] if r % 2 else ["proto", "prod"]
    line = []
    for name in order:
        gr = graphs[name]
        pre = live()
        us = b.timeit(lambda gr=gr: gr.replay())
        vals[name].append(us)
        line.append("%s=%.2f [%s -> %s]" % (name, us, pre, live()))
    print("r%-2d %s" % (r, "  ".join(line)), flush=True)
    time.sleep(3)

lo, hi = sorted([statistics.median(vals["prod"]), statistics.median(vals["proto"])])
print("MEDIAN prod=%.2f proto=%.2f  spread=%.2f  prod-proto=%+.2f us"
      % (statistics.median(vals["prod"]), statistics.median(vals["proto"]),
         max(vals["prod"] + vals["proto"]) - min(vals["prod"] + vals["proto"]),
         statistics.median(vals["prod"]) - statistics.median(vals["proto"])),
      flush=True)
allp = sorted(vals["prod"])
print("prod sorted=%s" % allp, flush=True)
print("BISTABILITY-DONE", flush=True)
