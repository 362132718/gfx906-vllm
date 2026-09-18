#!/usr/bin/env python3
"""补丁回归验证（在主机 x@192.168.10.11 上跑，只读 + 两次 HTTP 请求）。

覆盖：
  1. 启动日志：ROCm FP8 posture / KV 容量 / 并发度 / 有无 WARNING-ERROR
  2. 文本 smoke
  3. 图像 smoke（多模态）
  4. 前缀缓存命中率（调既有 prefix_hit_probe.sh，量化 PREFIX_MATCH_UNIT=16 是否仍生效）
  5. 显存占用
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = open("/home/x/.vllm-api-key").read().strip()
LOG = "/home/x/logs/serve.log"
HDRS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers=HDRS
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
            return r.status, body, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600], time.time() - t0


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


section("1. 启动日志证据")
keys = (
    "ROCm FP8 posture",
    "GPU KV cache size",
    "Maximum concurrency",
    "Available KV cache memory",
    "Setting attention block size",
    "prefix_match_unit",
    "kv_cache_dtype",
    "max_model_len",
    "ERROR",
    "ValueError",
    "Traceback",
)
try:
    for i, ln in enumerate(open(LOG, encoding="utf-8", errors="replace"), 1):
        if any(k in ln for k in keys):
            print(f"{i:6d}| {ln.rstrip()[:200]}")
except FileNotFoundError:
    print(f"!! {LOG} 不存在")


def smoke(name, path):
    section(f"{name} smoke")
    try:
        payload = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"!! 读不到 {path}: {exc}")
        return
    code, body, dt = post("/v1/chat/completions", payload)
    if code != 200:
        print(f"HTTP {code} in {dt:.2f}s -> {body}")
        return
    msg = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(
        f"HTTP 200  {dt:.2f}s  prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')}"
    )
    print("--- 回答 ---")
    print(msg[:500])


smoke("2. 文本", "/home/x/mm_text.json")
smoke("3. 图像（多模态）", "/home/x/mm_image.json")

section("4. 前缀缓存命中率（PREFIX_MATCH_UNIT=16 是否仍生效）")
try:
    out = subprocess.run(
        ["bash", "/home/x/prefix_hit_probe.sh", "patch后"],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd="/home/x",
    )
    print(out.stdout)
    if out.stderr.strip():
        print("[stderr]", out.stderr[:800])
except subprocess.TimeoutExpired:
    print("!! prefix_hit_probe.sh 超时")

section("5. 显存与功耗")
for cmd in (
    ["rocm-smi", "--showmeminfo", "vram", "--showuse"],
    ["rocm-smi", "--showpower", "--showtemp"],
):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(r.stdout)
    except Exception as exc:  # noqa: BLE001
        print(f"!! {cmd[0]} 失败: {exc}")

print("\n[verify_patch_2] DONE")
sys.exit(0)
