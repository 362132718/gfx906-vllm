#!/usr/bin/env python3
"""Minimal vLLM OpenAI-API smoke test (stdlib only).

Fires one chat completion with a unique nonce so the prefix cache cannot
serve it, then reports latency and decode speed.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.time() - t0


def main():
    models = get("/v1/models")
    ids = [m["id"] for m in models["data"]]
    print(f"models: {ids}", flush=True)
    model = ids[0]

    nonce = f"smoke-{int(time.time())}"
    prompt = (
        f"[{nonce}] 请用三句话说明 Gated DeltaNet 与标准注意力在 KV cache 上的区别。"
    )

    body, dt = post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.0,
        },
    )
    usage = body.get("usage", {})
    txt = body["choices"][0]["message"]["content"]
    ct = usage.get("completion_tokens", 0)
    print(f"latency={dt:.2f}s completion_tokens={ct} -> {ct / dt:.2f} tok/s", flush=True)
    print("--- answer head ---", flush=True)
    print(txt[:400], flush=True)


if __name__ == "__main__":
    sys.exit(main())
