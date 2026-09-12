"""Show exactly how each P2 plan fence differs from the landed file.

p2_plan_selfcheck asserts "fence.strip() in landed" and stops at the first failure;
this prints the line-level diff so a DRIFT verdict can be read instead of guessed.
Anchor = first/last fence line matched after stripping, so a drift *inside* the
block still lands on the right window.

Run in the container:  python3 p2_fence_diff.py <plan.md> <landed.py>
"""
import difflib
import re
import sys

md, landed_path = sys.argv[1], sys.argv[2]
text = open(md, encoding="utf-8").read()
blocks = re.findall(r"```(?:cpp|python)\n(.*?)```", text, re.S)
sel = {
    "loader":       [b for b in blocks if "_P2_SHAPES = frozenset" in b],
    "registration": [b for b in blocks if "def _p2_w4a16_gemm_impl" in b],
    "warm-up":      [b for b in blocks if "self._p2_part = None" in b],
    "route":        [b for b in blocks if "_P2_HITS.append" in b],
}
for k, v in sel.items():
    assert len(v) == 1, "%s: %d blocks" % (k, len(v))
L = open(landed_path, encoding="utf-8").read().split("\n")
Ls = [l.strip() for l in L]

for name, blk in sel.items():
    b = blk[0].rstrip("\n").split("\n")
    bs = [l.strip() for l in b]
    start = None
    for i in range(len(Ls)):
        if Ls[i] == bs[0] and i + len(bs) <= len(Ls) and Ls[i + 1:i + len(bs)] == bs[1:]:
            start = i
            break
    if start is None:                      # first line alone drifted
        for i in range(len(Ls)):
            if Ls[i] == bs[0]:
                start = i
                break
    if start is None:                      # ...so did the last
        for i in range(len(Ls)):
            if Ls[i] == bs[-1]:
                start = max(0, i - len(bs) + 1)
                break
    if start is None:
        print("#### %s: fence not found in landed file at all" % name)
        continue
    win = L[start:start + len(bs) + 4]
    identical = win[:len(bs)] == b
    print("#### %-12s landed line %-5d %s" % (name, start + 1,
          "IDENTICAL" if identical else "DIFFERS"))
    if not identical:
        for line in difflib.unified_diff(b, win, "fence", "landed", lineterm="", n=1):
            print("   " + line)
