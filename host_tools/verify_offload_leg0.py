#!/usr/bin/env python3
"""腿 0 冒烟验证：tiering offload 能不能在本模型上起来 + 指标面是否出现。

在主机 x@192.168.10.11 上跑：ssh x@... 'python3 -' < verify_offload_leg0.py
只读日志 + 两个 HTTP 请求 + 数一下 tier 目录。不改任何配置。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = open("/home/x/.vllm-api-key").read().strip()
LOG = "/home/x/logs/serve.log"
TIER_DIR = "/home/x/logs/kv-tier"
HDRS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


section("1. 日志取证（H1 的四个判定点）")
KEYS = (
    "mmap",
    "Opened existing",
    "Created mmap file",
    "OffloadingConnector",
    "TieringOffloadingSpec",
    "Turning off hybrid kv cache manager",
    "does not support HMA",
    "TimeoutError",
    "ValueError",
    "Error",
    "GPU KV cache size",
    "Maximum concurrency",
    "Available KV cache memory",
    "Setting attention block size",
    "kv_offload",
)
try:
    for i, ln in enumerate(open(LOG, encoding="utf-8", errors="replace"), 1):
        if any(k in ln for k in KEYS):
            print(f"{i:6d}| {ln.rstrip()[:210]}")
except FileNotFoundError:
    print(f"!! {LOG} 不存在")


def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers=HDRS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


section("2. /metrics 里的 offload 指标面（发现全名）")
try:
    m0 = metrics()
    off = sorted({ln.split(" ")[0].split("{")[0] for ln in m0.split("\n") if "offload" in ln})
    print(f"命中 {len(off)} 个指标名：")
    for n in off:
        print("   ", n)
    if not off:
        print("   (无 —— 说明 offload 未被引擎注册)")
except Exception as exc:  # noqa: BLE001
    print("!! /metrics 读取失败:", exc)


def post(prompt, max_tokens=32, timeout=900):
    payload = {
        "model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(), headers=HDRS
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        return r.status, body, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500], time.time() - t0


section("3. 打一个 ~2K token 的请求，看 store 路径是否真的被触发")
filler = "这是用于触发 KV 块产生与卸载的中性填充文本，内容本身无意义。"
prompt = filler * 140 + "\n请只回答一个词：收到"
print(f"prompt 字符数 = {len(prompt)}")
code, body, dt = post(prompt)
print(f"HTTP {code}  {dt:.2f}s")
if code == 200:
    print(f"usage = {body.get('usage')}")
    print(f"answer head = {body['choices'][0]['message']['content'][:120]!r}")
else:
    print("body:", body)

time.sleep(5)
section("4. 请求后的指标增量 + tier 目录落盘")
try:
    m1 = metrics()
    for ln in m1.split("\n"):
        if "offload" in ln and not ln.startswith("#"):
            print("   ", ln[:150])
except Exception as exc:  # noqa: BLE001
    print("!! /metrics 读取失败:", exc)

print("\n--- tier 目录 ---")
print("du:", sh(f"du -sh {TIER_DIR}"))
print("文件数:", sh(f"find {TIER_DIR} -type f | wc -l"))
print("capability 文件:", sh(f"find {TIER_DIR} -maxdepth 2 | head -12"))
print("\n--- /dev/shm 里的 mmap ---")
print(sh('docker exec vllm bash -lc "ls -l /dev/shm/vllm_offload_*.mmap 2>/dev/null || echo none"'))

print("\n--- GPU 显存 ---")
print(sh("rocm-smi --showmeminfo vram 2>/dev/null | grep Used"))
print("\n[leg0] DONE")
sys.exit(0)
