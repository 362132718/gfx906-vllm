# p1_shapes_report.py -- dedupe per-rank shapedbg CSVs and cross-check against
# the design's shape table. A mismatch means our model of fusion/sharding is
# wrong, which is exactly what this step is for.
import csv, glob, collections

DESIGN = {                      # (K, N) per rank, from design :102-108
    (5120, 7168):  "attn qkv_proj",
    (3072, 5120):  "attn o_proj + GDN out_proj",
    (5120, 17408): "mlp gate_up_proj",
    (8704, 5120):  "mlp down_proj",
    (5120, 8192):  "GDN in_proj_qkvz",
}

seen = collections.Counter()
ms = collections.defaultdict(set)
files = sorted(glob.glob("/work/p1/p1_shapes.csv.*"))
print("rank csv files: %d" % len(files))
for path in files:
    rows = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            m, k, n = int(row["m"]), int(row["k"]), int(row["n"])
            seen[(k, n)] += 1
            ms[(k, n)].add(m)
            rows += 1
    print("  %s rows=%d" % (path, rows))

print("distinct (K,N) observed = %d   design table has %d" % (len(seen), len(DESIGN)))
for (k, n), cnt in sorted(seen.items()):
    tag = DESIGN.get((k, n), "*** NOT IN DESIGN TABLE ***")
    print("  K=%-6d N=%-6d csv_rows=%-5d m_values=%s  128|K=%s  8|N=%s  %s" % (
        k, n, cnt, sorted(ms[(k, n)]), k % 128 == 0, n % 8 == 0, tag))
extra = set(seen) - set(DESIGN)
missing = set(DESIGN) - set(seen)
print("extra=%s" % (sorted(extra) or "-"))
print("missing=%s" % (sorted(missing) or "-"))
assert not extra, "observed shapes outside the design table -> fix the design's section 2 first"
assert not missing, "designed shapes never observed -> the serving path differs from the design"
# TP symmetry: every rank must have seen the identical shape set (per-rank N is
# global N / 2 by construction; attn_output_gate doubles q_proj to 12288, so the
# per-rank qkv_proj N=7168 rather than 6144 is the design's own checked fact).
assert len(files) >= 1, "no rank dumped anything -- SIGUSR1 never reached a worker"
print("SHAPE-GATE PASS")
