# a2_client.py -- streaming long-context benchmark per the Z100 author's
# protocol: temperature=0, 64 output tokens, ignore_eos, prefill tok/s from
# TTFT, decode tok/s from first-to-last token. 3 trials per length; trial 0
# is the cold run and is reported separately.
import json
import random
import sys
import time
import urllib.error
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
URL = "http://127.0.0.1:%d/v1/completions" % PORT
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango "
         "uniform victor whiskey xray yankee zulu").split()


def make_prompt(n_tokens, seed):
    rnd = random.Random(seed)
    # this tokenizer yields ~1.54 tokens per filler word; API-reported
    # prompt_tokens drives the rate math regardless of estimate error.
    return " ".join(rnd.choice(WORDS) for _ in range(int(n_tokens / 1.54)))


def stream_once(prompt):
    payload = {"model": "qwen38", "prompt": prompt, "max_tokens": 64,
               "temperature": 0.0, "stream": True, "ignore_eos": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            d = json.loads(body)
            if d.get("usage"):
                prompt_tokens = d["usage"].get("prompt_tokens")
            for ch in d.get("choices", []):
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.time() - t0
                    ntok += len(ch.get("token_ids") or [1])
    t_end = time.time() - t0
    return ttft or (t_end - t0), t_end, ntok, prompt_tokens or 0


def main():
    results = []
    for target in (2048, 8192, 16320):
        for trial in range(3):
            prompt = make_prompt(target, trial * 97 + target)
            try:
                ttft, tot, out_tok, p_tok = stream_once(prompt)
            except urllib.error.HTTPError as e:
                print("target=%d trial=%d HTTP %d, retrying 10%% shorter"
                      % (target, trial, e.code), flush=True)
                prompt = make_prompt(int(target * 0.9), trial * 97 + target)
                ttft, tot, out_tok, p_tok = stream_once(prompt)
            pre = p_tok / ttft if ttft > 0 else 0.0
            dec = (out_tok - 1) / (tot - ttft) if tot - ttft > 0.05 else 0.0
            line = ("target=%d prompt_tokens=%d out=%d TTFT=%.2fs "
                    "prefill=%.1f tok/s decode=%.2f tok/s"
                    % (target, p_tok, out_tok, ttft, pre, dec))
            print(("[COLD] " if trial == 0 else "") + line, flush=True)
            results.append((target, trial, p_tok, out_tok, ttft, tot, pre, dec))
        print("-" * 60, flush=True)
    print("SUMMARY (warm trials only):")
    for target in (2048, 8192, 16320):
        warm = [r for r in results if r[0] == target and r[1] > 0]
        if warm:
            print("  %6d: prefill %.1f tok/s   decode %.1f tok/s"
                  % (target, sum(r[6] for r in warm) / len(warm),
                     sum(r[7] for r in warm) / len(warm)))


if __name__ == "__main__":
    main()
