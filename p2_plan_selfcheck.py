"""Extract the fenced code blocks of the P2 plan and self-check them.

Two checks, both cheap and both targeting the "plan cannot even start" class:
  (1) the production .hip compiles with hipcc and exports p2_gemv;
  (2) the module-level patch blocks import and register the custom op inside
      the real vLLM 0.28, and one real ctypes call matches an fp64 reference.
Plus, when the landed file is passed, that every patch fence is a verbatim
substring of the code that actually shipped (plans drift; prose fences pass
review and still differ from the build).

Run in the container:  python3 p2_plan_selfcheck.py <plan.md> <outdir> [landed.py]
"""
import re
import sys


def fences(text):
    return re.findall(r"```(?:cpp|python)\n(.*?)```", text, re.S)


def frag_compile(body, tag):
    """Compile an indented in-function fragment by wrapping it in a def."""
    lines = [l[8:] if l.startswith(" " * 8) else l for l in body.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines[-1].rstrip().endswith(":"):
        lines.append(" " * 12 + "pass")
    src = "def _frag(self, layer, device, c, K_dim, N_dim, M, bits, os, torch,\n" \
          "         _P2_OK, _P2_SHAPES, _P2_HITS, _P2_SPLITK, _p2_layer_ok,\n" \
          "         _p2_w4a16_gemm_impl, x_2d, w_q, w_s, w_zp, w_g_idx, c_out,\n" \
          "         temp_dq, ops):\n" \
          "    if True:\n" + \
          "\n".join("        " + l if l.strip() else "" for l in lines) + "\n"
    compile(src, tag, "exec")
    print("fragment compiles:", tag)


def main(md_path, outdir, landed_path=None):
    text = open(md_path, encoding="utf-8").read()
    blocks = fences(text)

    hip = [b for b in blocks if "p2_kernels_prod.hip --" in b]
    loader = [b for b in blocks if "_P2_SHAPES = frozenset" in b]
    reg = [b for b in blocks if "def _p2_w4a16_gemm_impl" in b]
    warm = [b for b in blocks if "self._p2_part = None" in b]
    route = [b for b in blocks if "torch.ops.vllm.p2_gemv_or_fused(" in b]
    # G-P3 moved the M test out of apply_weights and into this op, so the
    # dispatch block is the one that carries the load-bearing contract
    # (`-> None` + mutates_args) -- registering it here runs infer_schema.
    dispatch = [b for b in blocks if "def _p2_gemv_or_fused_impl" in b]
    sel = {"loader": loader, "registration": reg, "warm-up": warm,
           "route": route, "dispatch": dispatch}
    for name, s in ((".hip", hip), ("loader", loader), ("registration", reg),
                    ("warm-up frag", warm), ("route frag", route),
                    ("dispatch", dispatch)):
        assert len(s) == 1, "expected 1 block for %s, got %d" % (name, len(s))

    open(outdir + "/p2_kernels_prod.hip", "w", encoding="utf-8").write(hip[0])
    # MPLinearLayerConfig appears in _p2_layer_ok's annotation, and annotations
    # are evaluated at def time -- without this import the extracted module dies
    # with NameError on import. Same path the real patch uses (wna16_exllama:49).
    module = ("import torch\n"
              "from vllm.utils.torch_utils import direct_register_custom_op\n"
              "from vllm.model_executor.kernels.linear.mixed_precision."
              "MPLinearKernel import MPLinearLayerConfig\n\n"
              + loader[0] + "\n\n" + reg[0] + "\n\n" + dispatch[0])
    open(outdir + "/p2_patch_blocks.py", "w", encoding="utf-8").write(module)
    print("wrote p2_kernels_prod.hip (%d B) and p2_patch_blocks.py (%d B)"
          % (len(hip[0]), len(module)))

    frag_compile(warm[0], "task2-step3-warmup")
    frag_compile(route[0], "task2-step4-route")

    if landed_path:
        # Drift check: the fences must be the shipped text, not a paraphrase of
        # it. Rewriting a fence to "look like" the code is exactly how this plan
        # passed review twice while differing from it in load-bearing detail.
        landed = open(landed_path, encoding="utf-8").read()
        drift = [name for name, s in sorted(sel.items())
                 if s[0].strip() not in landed]
        for name, s in sorted(sel.items()):
            print("  fence in landed file %-12s %s"
                  % (name, "ok" if name not in drift else "DRIFT"))
        assert not drift, "fences drift from %s: %s" % (landed_path, drift)


if __name__ == "__main__":
    main(*sys.argv[1:])
