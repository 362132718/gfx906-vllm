# p05_accuracy.py -- LLMM1 vs Tensile vs an fp64 reference on the REAL weights.
#   ssh x@192.168.10.11 'docker exec -i vllm-gfx906 python3 /work/p0/p05_accuracy.py'
# Per-row dot products do not care about row order, so the separate
# in_proj_a checkpoint tensor is a faithful operand for the fused in_proj_ba.
import glob, json, os, struct
import torch
from safetensors import safe_open            # vllm's own dependency
from vllm import _custom_ops as ops

MODEL = os.environ.get("MODEL",
                       "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
N_ROWS_LM_HEAD = 8192      # the full 124160-row fp64 reference needs ~5 GB
N_X = 64
OUT = os.environ.get("OUT", "/work/p0")
torch.manual_seed(1234)


def locate(suffix):
    """(shard_path, key) for the first tensor whose name ends with `suffix`.
    Reads only the 8-byte + JSON header of each shard -- no 2.5 GB load.
    Suffixes carry no leading dot on purpose: the checkpoint stores the head as
    the bare key "lm_head.weight", so ".lm_head.weight" would match nothing."""
    for shard in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
        with open(shard, "rb") as f:
            hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for k in hdr:
            if k.endswith(suffix):
                return shard, k
    raise SystemExit("FATAL: no tensor ending in %r under %s" % (suffix, MODEL))


def load_rows(shard, key, n_rows=None):
    """All rows, or n_rows spread over the whole first axis. A prefix slice of
    lm_head would take the vocab's leading control/space tokens and bias the
    top-1 margin statistics this gate exists to measure."""
    with safe_open(shard, framework="pt") as f:
        t = f.get_tensor(key)
    if n_rows and t.shape[0] > n_rows:
        t = t[torch.linspace(0, t.shape[0] - 1, n_rows).long()]
    return t.contiguous()


def grade(tag, w_cpu, xs):
    w = w_cpu.to(torch.float16).cuda().contiguous()
    k = w.shape[1]
    rel_g, rel_t, flips, margins = [], [], 0, []
    for x in xs:
        x = (x * 0.55).cuda().half().view(1, k)      # a decode hidden state is
        ref = torch.sum(w.double() * x.double(), dim=1)      # not unit-norm
        g = ops.LLMM1(w, x, 4).view(-1).double()
        t = torch.nn.functional.linear(x, w).view(-1).double()
        rel_g.append(((g - ref).abs().max() / ref.abs().max()).item())
        rel_t.append(((t - ref).abs().max() / ref.abs().max()).item())
        if w.shape[0] > 8:
            flips += int(g.argmax().item() != t.argmax().item())
            s = g.sort(descending=True).values
            margins.append(float((s[0] - s[1]).abs().item() / s[0].abs().item()))
    r = {"case": tag, "m": w.shape[0], "k": k,
         "rel_gemv_max": round(max(rel_g), 6),
         "rel_tensile_max": round(max(rel_t), 6),
         "ratio": round(max(rel_g) / max(max(rel_t), 1e-12), 2),
         "argmax_flips": "%d/%d" % (flips, len(xs)),
         "top1_margin_min": round(min(margins), 5) if margins else None}
    print(json.dumps(r), flush=True)
    return r


xs = lambda k: [torch.randn(k) for _ in range(N_X)]
shard, key = locate("linear_attn.in_proj_a.weight")
res = [grade("in_proj_ba[%s]" % key.rsplit(".", 2)[-2],
             load_rows(shard, key), xs(k=5120))]
shard, key = locate("lm_head.weight")
res.append(grade("lm_head[%d/%d rows]" % (N_ROWS_LM_HEAD, 248320),
                 load_rows(shard, key, N_ROWS_LM_HEAD), xs(k=5120)))

for r in res:
    assert r["ratio"] <= 3.0, \
        "%(case)s: skinny GEMV is %(ratio)sx Tensile's fp64 error (>3x) -> not a" \
        " drop-in, stop and revisit the kernel choice" % r
    assert r["rel_gemv_max"] < 5e-3, "%s: absolute error too large" % r["case"]
os.makedirs(OUT, exist_ok=True)
json.dump(res, open("%s/p05_accuracy.json" % OUT, "w"), indent=1)
print("ALL PASS -> %s/p05_accuracy.json" % OUT)
