"""lin_census_hook.py -- per-process census of torch.nn.functional.linear.

Installed as a `.pth`-activated module so that it loads in EVERY python process,
including vLLM's spawned EngineCore and Worker_TP* processes. That is the whole
point: with TP=2 the model forward runs in worker subprocesses, so a
monkeypatch applied in the parent (the obvious approach) sees exactly zero
calls -- measured 2026-09-11.

Goal: name the fp16 linears that fall through to rocBLAS instead of the gfx906
P0.5 skinny GEMV path (docker/patches/layers_utils.py:209-219), which in the
delivery-mode profile cost 6.03 ms/step across 40 calls/forward.

Activated only when LIN_CENSUS_OUT is set, so it is inert otherwise.
Dumps every DUMP_EVERY calls (not only at exit) because vLLM terminates workers
with a queue signal and atexit is not guaranteed to run.
"""
import atexit
import collections
import json
import os
import traceback

_OUT = os.environ.get("LIN_CENSUS_OUT")
DUMP_EVERY = int(os.environ.get("LIN_CENSUS_EVERY", "200"))

if _OUT:
    import torch
    import torch.nn.functional as _F

    _orig = _F.linear
    _stats = collections.Counter()
    _stacks = {}
    _n = [0]

    def _key(x, weight, bias):
        n = x.numel() // x.size(-1) if x.numel() else 0
        return (n, int(weight.shape[0]), int(weight.shape[1]),
                str(x.dtype).replace("torch.", ""), bias is not None,
                bool(weight.is_contiguous()), bool(x.is_contiguous()))

    def _dump():
        try:
            rows = []
            for k, v in _stats.items():
                rows.append({"n": k[0], "m": k[1], "k": k[2], "dtype": k[3],
                             "has_bias": k[4], "w_contig": k[5],
                             "x_contig": k[6], "calls": v,
                             "stack": list(_stacks.get(k, []))})
            tmp = "%s.pid%d.json" % (_OUT, os.getpid())
            with open(tmp + ".tmp", "w") as fh:
                json.dump({"pid": os.getpid(), "rows": rows}, fh, indent=1)
            os.replace(tmp + ".tmp", tmp)      # atomic; never a half file
        except Exception:  # noqa: BLE001
            pass

    def _patched(x, weight, bias=None, *a, **kw):
        if isinstance(weight, torch.Tensor):
            try:
                k = _key(x, weight, bias)
                _stats[k] += 1
                if k not in _stacks:
                    fr = [f for f in traceback.extract_stack()[:-1]
                          if "/vllm/" in f.filename]
                    _stacks[k] = ["%s:%d %s" % (os.path.basename(f.filename),
                                                f.lineno, f.name)
                                  for f in fr[-4:]]
                _n[0] += 1
                if _n[0] % DUMP_EVERY == 0:
                    _dump()
            except Exception:  # noqa: BLE001
                pass
        return _orig(x, weight, bias, *a, **kw)

    _F.linear = _patched
    atexit.register(_dump)
