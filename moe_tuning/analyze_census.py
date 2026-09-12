import collections
import glob
import json

agg = collections.Counter()
stacks = {}
for f in sorted(glob.glob("/home/x/vllm-work/p5/census/*.json")):
    d = json.load(open(f))
    for r in d.get("rows", []):
        k = (r["n"], r["m"], r["k"], r["dtype"], r["has_bias"],
             r["w_contig"], r["x_contig"])
        agg[k] += r["calls"]
        if k not in stacks:
            stacks[k] = r.get("stack", [])


def p05(k):
    n, m, kk, dt, hb, wc, xc = k
    return (dt in ("float16", "bfloat16") and not hb and n == 1
            and m % 4 == 0 and kk % 8 == 0 and kk <= 8192 and wc)


def why(k):
    n, m, kk, dt, hb, wc, xc = k
    w = []
    if hb:
        w.append("bias!=None")
    if not wc:
        w.append("w_noncontig")
    if n != 1:
        w.append("n=%d" % n)
    if kk > 8192:
        w.append("k>8192")
    if kk % 8:
        w.append("k%%8=%d" % (kk % 8))
    if m % 4:
        w.append("m%%4=%d" % (m % 4))
    if dt not in ("float16", "bfloat16"):
        w.append(dt)
    return ",".join(w) or "?"


for label, sel in (("全部", lambda k: True), ("decode (n==1)", lambda k: k[0] == 1)):
    sub = {k: c for k, c in agg.items() if sel(k)}
    tot = sum(sub.values())
    ok = {k: c for k, c in sub.items() if p05(k)}
    bad = {k: c for k, c in sub.items() if not p05(k)}
    print("=" * 78)
    print("【%s】调用 %d 次 / %d 种形状   ->  P0.5 命中 %d 次 (%.0f%%)，落 rocBLAS %d 次"
          % (label, tot, len(sub), sum(ok.values()),
             100.0 * sum(ok.values()) / max(tot, 1), sum(bad.values())))
    print("=" * 78)
    print("%5s %8s %7s %-8s %5s %5s %7s  %s"
          % ("n", "m", "k", "dtype", "bias", "wcont", "calls", "verdict"))
    for k, c in sorted(sub.items(), key=lambda kv: -kv[1])[:22]:
        print("%5d %8d %7d %-8s %5s %5s %7d  %s"
              % (k[0], k[1], k[2], k[3], k[4], k[5], c,
                 "P0.5 ok" if p05(k) else "-> rocBLAS (" + why(k) + ")"))
    print()
    if bad:
        print("  逃逸项的调用栈:")
        for k, c in sorted(bad.items(), key=lambda kv: -kv[1])[:4]:
            print("    n=%d m=%d k=%d x%d :" % (k[0], k[1], k[2], c))
            for s in stacks.get(k, [])[:4]:
                print("       ", s)
        print()
