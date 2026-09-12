# p2_l3_ctrl.py -- read-side only: how big is the noise where NOTHING was swapped?
#
# The gate floors M0_swap against off-arm prefill-vs-its-own-decode. The cross-arm
# PREFILL comparison is a purer control: at M>1 neither arm takes the M==1 branch, so
# any flip there is the engine's own run-to-run jitter (the incumbent fused kernel
# reduces with atomicAdd). If that control's worst margin is as large as the swap's,
# the pass does not depend on which noise definition the floor used.
import p2_l3_gate as G

off, on = G.load("off"), G.load("on")
ctrl, swap, floor = [], [], []
n_dec = n_pre = 0
for pi, (ro, rn) in enumerate(zip(off["rows"], on["rows"])):
    pf_o, pf_n = G.by_pos(ro["prefill"]), G.by_pos(rn["prefill"])
    df_o, df_n = G.by_pos(ro["decode_forced"]), G.by_pos(rn["decode_forced"])
    n_pre += len(set(pf_o) & set(pf_n))
    n_dec += len(set(df_o) & set(df_n))
    for src, dst, tag in ((pf_o, pf_n, ctrl), (df_o, df_n, swap), (pf_o, df_o, floor)):
        for i in sorted(set(src) & set(dst)):
            f = G.flip(src[i], dst[i], i, i)
            if f:
                tag.append((pi, f))


def show(tag, xs):
    print("%-6s n=%d worst=%.4f  %s" % (
        tag, len(xs), max((f[1] for _, f in xs), default=0.0),
        [(pi, f[0], round(f[1], 4), f[2], f[3]) for pi, f in
         sorted(xs, key=lambda t: -t[1][1])[:8]]), flush=True)


show("swap", swap)
show("floor", floor)
show("ctrl", ctrl)
band = [f[1] for _, f in floor] + [f[1] for _, f in ctrl]
print("shared decode pos=%d shared prefill pos=%d" % (n_dec, n_pre), flush=True)
print("swap_worst=%.4f <= max(floor,ctrl)=%.4f -> %s" % (
    max([f[1] for _, f in swap] + [0.0]), max(band + [0.0]),
    max([f[1] for _, f in swap] + [0.0]) <= max(band + [0.0])
    + 1e-12), flush=True)
