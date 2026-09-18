#!/usr/bin/env python3
"""腿 0 补充测量：offload 的**每 token 落盘/搬运量**。

为什么测：本次 2720-token 请求产生了 636 MB 的 store + fs 写入 —— 比 KV 池公式
(24.1 KB/token) 大约 6×。A/B 要用 ~50K token 的前缀，若真是这个量级就是几 GB/次，
必须先把"每 token 多少字节"量准，否则 H3 的代价算错。
做法：发两个**不同**的 prompt（不同 filler，避免前缀命中），各自读 store_bytes 增量。
"""
import json
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = open("/home/x/.vllm-api-key").read().strip()
HDRS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
M = ("vllm:kv_offload_store_bytes_total", "vllm:kv_offload_tiering_write_bytes_total")


def metric(name):
    req = urllib.request.Request(BASE + "/metrics", headers=HDRS)
    with urllib.request.urlopen(req, timeout=60) as r:
        for ln in r.read().decode("utf-8", "replace").split("\n"):
            if ln.startswith(name + "{") and "gpu" not in ln:
                try:
                    return float(ln.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return None


def snap():
    return {n: metric(n) for n in M}


def ask(tag, mult):
    filler = f"【{tag}】这是用于测量卸载量的填充文本，语义无关。" * mult
    body = {
        "model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
        "messages": [{"role": "user", "content": filler + "\n只回答：好"}],
        "max_tokens": 4,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HDRS
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["usage"]["prompt_tokens"], time.time() - t0


def files():
    out = subprocess.run(
        "find /home/x/logs/kv-tier -type f | wc -l", shell=True, capture_output=True, text=True
    ).stdout.strip()
    du = subprocess.run(
        "du -sm /home/x/logs/kv-tier | cut -f1", shell=True, capture_output=True, text=True
    ).stdout.strip()
    return out, du


print(f"{'tag':>10} {'prompt_tok':>10} {'dt(s)':>7} {'ΔGPU→CPU(MB)':>14} {'Δfs写入(MB)':>13} {'KB/token':>9}")
prev = snap()
for tag, mult in (("F1", 70), ("F2", 140), ("F3", 350)):
    tok, dt = ask(tag, mult)
    time.sleep(6)
    cur = snap()
    d_store = (cur[M[0]] or 0) - (prev[M[0]] or 0)
    d_fs = (cur[M[1]] or 0) - (prev[M[1]] or 0)
    per = (d_store / tok / 1024) if tok else 0
    print(
        f"{tag:>10} {tok:>10} {dt:>7.2f} {d_store/1e6:>14.1f} {d_fs/1e6:>13.1f} {per:>9.1f}",
        flush=True,
    )
    prev = cur

n, mb = files()
print(f"\nfs 层现状: {n} 个文件, {mb} MB")
print("参考: KV 池公式给的是 ≈24.1 KB/token（fp16，attention+mamba 合计）")
