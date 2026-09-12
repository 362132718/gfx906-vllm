# a2b_client.py -- long-context DECODE probe on the delivery serve path.
# argv: <port> <model_id> [targets]
# decode tok/s is measured from first token to last token of a 64-token greedy
# request whose prompt fills the context, so the KV that attention reads is
# ~prompt-length for the whole measured window.  trial 0 is the cold run.
import json
import random
import sys
import time
import urllib.error
import urllib.request

PORT = int(sys.argv[1])
MODEL = sys.argv[2]
TARGETS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                            else ["2048", "8192", "16256"])]
OUT_TOK = 64
URL = "http://127.0.0.1:%d/v1/completions" % PORT
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango "
         "uniform victor whiskey xray yankee zulu").split()


def make_prompt(n_tokens, seed):
    rnd = random.Random(seed)
    # ~1.54 tokens per filler word with this tokenizer; the API-reported
    # prompt_tokens drives all rate math so the estimate only needs to be close.
    return " ".join(rnd.choice(WORDS) for _ in range(int(n_tokens / 1.54)))


def stream_once(prompt):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": OUT_TOK,
               "temperature": 0.0, "stream": True, "ignore_eos": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    text = []
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                    text.append(ch["text"])
    return ttft or (time.time() - t0), time.time() - t0, ntok, prompt_tokens or 0, "".join(text)


def main():
    rows = []
    for target in TARGETS:
        for trial in range(3):
            prompt = make_prompt(target, trial * 97 + target)
            try:
                ttft, tot, out, p_tok, txt = stream_once(prompt)
            except urllib.error.HTTPError as e:
                print("target=%d trial=%d HTTP %d, retrying 10%% shorter"
                      % (target, trial, e.code), flush=True)
                prompt = make_prompt(int(target * 0.9), trial * 97 + target)
                ttft, tot, out, p_tok, txt = stream_once(prompt)
            pre = p_tok / ttft if ttft > 0 else 0.0
            win = tot - ttft
            dec = (out - 1) / win if win > 0.05 else 0.0
            step_ms = 1000.0 / dec if dec > 0 else 0.0
            print("target=%d prompt_tokens=%d out=%d TTFT=%.2fs prefill=%.1f "
                    "decode=%.2f tok/s (%.2f ms/step) ctx_at_decode=%d text=%r"
                  % (target, p_tok, out, ttft, pre, dec, step_ms, p_tok,
                     txt[:48]), flush=True)
            rows.append((target, trial, p_tok, out, ttft, pre, dec))
        print("-" * 70, flush=True)
    print("A2B SUMMARY (warm trials only, mean of trials 1..2):")
    for target in TARGETS:
        warm = [r for r in rows if r[0] == target and r[1] > 0]
        if warm:
            print("  ctx~%6d:  prefill %7.1f tok/s   decode %6.2f tok/s   "
                  "decode %7.2f ms/step"
                  % (target,
                     sum(r[5] for r in warm) / len(warm),
                     sum(r[6] for r in warm) / len(warm),
                     1000.0 / (sum(r[6] for r in warm) / len(warm))))
    print("A2B_CLIENT_DONE")


if __name__ == "__main__":
    main()
