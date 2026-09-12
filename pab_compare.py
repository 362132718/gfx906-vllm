"""pab_compare.py -- summarise pab_run.sh arms.

  python3 pab_compare.py <dir>

Prints, per context length: decode tok/s for every arm, and the greedy
token-id equality against the FIRST arm. Token-id equality is part of the
acceptance rule -- an env-gated path that changes numbers without changing
tokens is a plain win; one that changes tokens needs a reason before it ships.
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/work/ab"
arms = []
for f in sorted(glob.glob(os.path.join(d, "*", "wall.json"))):
    try:
        arms.append(json.load(open(f)))
    except Exception as e:  # noqa: BLE001
        print(f"  !! unreadable {f}: {e}")

if not arms:
    print("  (no wall.json found)")
    sys.exit(0)

print("%-14s %8s %6s %12s %12s %10s" %
      ("arm", "load_s", "maxlen", "kv_dtype", "backend", "gdn_norm"))
for a in arms:
    print("%-14s %8.1f %6d %12s %12s %10s" %
          (a["label"], a["load_s"], a["max_len"],
           a.get("kv_cache_dtype") or "<default>", a["backend"],
           a.get("gdn_output_norm", "0")))

for a in arms:
    if a.get("kv_capacity_line"):
        print("  %-14s %s" % (a["label"], a["kv_capacity_line"]))

ctxs = sorted({r["ctx_requested"] for a in arms for r in a["rows"]})
print()
print("%-14s %6s %12s %14s %12s" %
      ("arm", "ctx", "prefill_tps", "decode_tps", "ms/step"))
for a in arms:
    for r in a["rows"]:
        print("%-14s %6d %12s %14.2f %12.2f" %
              (a["label"], r["ctx_requested"],
               ("%.1f" % r["prefill_tps"]) if r.get("prefill_tps") else "n/a",
               r["decode_tps"], r["decode_ms_per_step"]))

print()
print("== greedy token-id equality vs first arm (%s) ==" % arms[0]["label"])
base = {r["ctx_requested"]: r["token_ids"] for r in arms[0]["rows"]}
for a in arms[1:]:
    for r in a["rows"]:
        b = base.get(r["ctx_requested"])
        if b is None:
            print("  %-14s ctx %-6d no counterpart" % (a["label"], r["ctx_requested"]))
            continue
        n = min(len(b), len(r["token_ids"]))
        same = b[:n] == r["token_ids"][:n]
        first_diff = next((i for i in range(n) if b[i] != r["token_ids"][i]), None)
        print("  %-14s ctx %-6d tokens_equal=%-5s (len %d/%d, first_diff=%s)"
              % (a["label"], r["ctx_requested"], same, len(r["token_ids"]), len(b),
                 first_diff))
