import re

raw = open("/home/x/vllm-work/moe_tuning/tune-run.log", "rb").read().decode("utf8", "ignore")
raw = raw.replace("\r", "\n")

pat = re.compile(r"\(pid=(\d+)\)\s*:?\s*(\d+)%\|[^|]*\|\s*([\d.]+)k?/([\d.]+)k?\s*\[(\d+):(\d+)<")

def num(s):
    return float(s) * 1000 if False else float(s)

bars = {}
for m in pat.finditer(raw):
    pid, pct, cur, tot, mm, ss = m.groups()
    cur = float(cur) * (1000 if False else 1)
    tot = float(tot)
    key = (pid, tot)
    el = int(mm) * 60 + int(ss)
    prev = bars.get(key)
    if prev is None or cur > prev[0]:
        bars[key] = (cur, tot, el)

print("观测到的批次（每个 tqdm 总量 = 一个 batch size）:")
print("%6s %10s %10s %10s %10s" % ("pid", "total", "cur", "elapsed", "s/cfg"))
tot_cfg = 0.0
for (pid, tot), (cur, _, el) in sorted(bars.items(), key=lambda kv: kv[1][1]):
    rate = el / cur if cur else 0
    tot_cfg += tot
    print("%6s %10.0f %10.0f %9dm%02ds %10.2f" % (pid, tot, cur, el // 60, el % 60, rate))

print()
print("已出现的批次总数（配置数合计）: %.0f" % tot_cfg)
print("注意: 7 个 batch size 应有 7 个不同的 tqdm 总量；出现几个就说明进行到第几个。")
