# gdn_verify.py -- A1: GDN kernel correctness verification for gfx906
# Naive fp32 torch references vs the Triton kernels vLLM actually runs
# for the Qwen3.5/3.8 GDN layers. Independent of the kernels under test.
#
# Run inside container:  python3 -u /work/gdn_verify.py
import math
import sys

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
)

DEV = "cuda"
DT = torch.float16
FAILS = []
RESULTS = []


def check(name, got, ref, atol=8e-3, rtol=2e-2):
    got = got.float()
    ref = ref.float()
    adiff = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-2)
    max_abs = adiff.max().item()
    max_rel = (adiff / denom).max().item()
    ok = bool((adiff <= atol + rtol * ref.abs()).all().item())
    nan = int(torch.isnan(got).sum().item())
    RESULTS.append((name, ok, max_abs, max_rel, nan))
    if not ok:
        FAILS.append(name)
    print("%-56s %s  max_abs=%.3e max_rel=%.3e nan=%d"
          % (name, "PASS" if ok else "FAIL", max_abs, max_rel, nan),
          flush=True)


def gating_params(HV, L, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A_log = torch.randn(HV, generator=g, device=DEV, dtype=torch.float32) * 0.5 - 2.0
    dt_bias = torch.randn(HV, generator=g, device=DEV, dtype=torch.float32) * 0.5
    a = torch.randn(L, HV, generator=g, device=DEV, dtype=DT)
    b = torch.randn(L, HV, generator=g, device=DEV, dtype=DT)
    return A_log, dt_bias, a, b


# ---------------- naive references (torch-native, no triton) ---------------

def ref_recurrent(q, k, v, g, beta, h0, scale):
    """q,k [T,H,K]; v [T,HV,V]; g,beta [T,HV] (log-decay, beta). h0 [HV,V,K].
    Returns o [T,HV,V], final state [HV,V,K]. fp32 loop."""
    H, HV = q.shape[1], v.shape[1]
    Kd, Vd = q.shape[2], v.shape[2]
    S = h0.clone().float()
    o = torch.empty(q.shape[0], HV, Vd, device=q.device, dtype=torch.float32)
    rep = HV // H
    qr = q.float().repeat_interleave(rep, dim=1)
    kr = k.float().repeat_interleave(rep, dim=1)
    for t in range(q.shape[0]):
        S = S * torch.exp(g[t]).view(HV, 1, 1)
        pred = torch.einsum('hvk,hk->hv', S, kr[t])
        u = (v[t].float() - pred) * beta[t].view(HV, 1)
        S = S + torch.einsum('hv,hk->hvk', u, kr[t])
        o[t] = torch.einsum('hvk,hk->hv', S, qr[t] * scale)
    return o, S


# ---------------- T1: causal_conv1d_fn -------------------------------------

def causal_conv1d_fn_ref(x, w, bias, initial_states, silu=True):
    """x [dim,T] single seq; w [dim,width]; init [dim,W-1] (right-aligned)."""
    W = w.shape[1]
    if initial_states is not None:
        xin = torch.cat([initial_states, x.float()], dim=1)
    else:
        xin = torch.cat([torch.zeros(x.shape[0], W - 1, device=x.device, dtype=torch.float32),
                         x.float()], dim=1)
    out = F.conv1d(xin.unsqueeze(0), w.float().unsqueeze(1), bias.float(), groups=w.shape[0])
    out = out[..., -x.shape[1]:]
    if silu:
        out = F.silu(out)
    return out.squeeze(0).to(x.dtype)


def test_conv1d_fn():
    W = 4
    for (dim, T, use_init) in [(64, 128, False), (64, 128, True), (96, 65, True), (256, 512, True)]:
        g = torch.Generator(device=DEV).manual_seed(dim * 1000 + T)
        x = torch.randn(1, dim, T, generator=g, device=DEV, dtype=DT)
        w = torch.randn(dim, W, generator=g, device=DEV, dtype=DT)
        bias = torch.randn(dim, generator=g, device=DEV, dtype=DT)
        init_gpu = torch.randn(dim, W - 1, generator=g, device=DEV, dtype=DT) if use_init else None
        # reference input states
        init_ref = init_gpu.clone().float() if use_init else None
        cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
        # kernel needs conv_states cache; give it a 2-line buffer, line 1 used
        conv_states = torch.zeros(2, dim, W - 1, device=DEV, dtype=DT)
        if use_init:
            conv_states[1] = init_gpu
        cache_indices = torch.tensor([1], dtype=torch.int32, device=DEV)
        has_initial = torch.tensor([use_init], dtype=torch.bool, device=DEV)
        xt = x.squeeze(0)  # [dim, T]
        out = causal_conv1d_fn(
            xt, w, bias, conv_states=conv_states, has_initial_state=has_initial,
            cache_indices=cache_indices, query_start_loc=cu, metadata=None)
        ref = causal_conv1d_fn_ref(xt, w, bias, init_ref)
        check("T1 conv1d_fn dim=%d T=%d init=%d" % (dim, T, use_init), out, ref)
        # state cache must hold last W-1 input tokens
        if T >= W - 1:
            check("T1c conv1d_fn state-cache dim=%d T=%d" % (dim, T),
                  conv_states[1], xt[:, -(W - 1):])


# ---------------- T2: causal_conv1d_update ---------------------------------

def test_conv1d_update():
    W = 4
    for (dim, B) in [(64, 1), (64, 8), (256, 8)]:
        g = torch.Generator(device=DEV).manual_seed(dim * 31 + B)
        x = torch.randn(B, dim, generator=g, device=DEV, dtype=DT)
        w = torch.randn(dim, W, generator=g, device=DEV, dtype=DT)
        bias = torch.randn(dim, generator=g, device=DEV, dtype=DT)
        states = torch.randn(B + 1, dim, W - 1, generator=g, device=DEV, dtype=DT)
        idx = torch.arange(1, B + 1, dtype=torch.int32, device=DEV)
        # naive: y = bias + sum_j w[:,j]*window[j], window = [state, x]
        window = torch.cat([states[idx.long()].float(), x.float().unsqueeze(-1)], dim=-1)  # [B,dim,W]
        ref = (window * w.float().unsqueeze(0)).sum(-1) + bias.float()
        ref = F.silu(ref).to(DT)
        ref_state = torch.cat([states[idx.long()][:, :, 1:], x.unsqueeze(-1)], dim=-1)
        out = causal_conv1d_update(x, states, w, bias, "silu",
                                   conv_state_indices=idx, validate_data=False)
        check("T2 conv1d_update dim=%d B=%d" % (dim, B), out, ref)
        # new state = roll
        check("T2c conv1d_update state dim=%d B=%d" % (dim, B), states[idx.long()], ref_state)


# ---------------- T3: fused_post_conv_prep ---------------------------------

def test_post_conv_prep():
    H, HV, K, V = 8, 24, 128, 128
    L = 200
    g = torch.Generator(device=DEV).manual_seed(3)
    qkv_dim = 2 * H * K + HV * V
    conv_out = torch.randn(L, qkv_dim, generator=g, device=DEV, dtype=DT)
    A_log, dt_bias, a, b = gating_params(HV, L, 4)
    q, k, v, ggo, beta = fused_post_conv_prep(
        conv_out, a, b, A_log, dt_bias, num_k_heads=H, head_k_dim=K, head_v_dim=V,
        apply_l2norm=True, output_g_exp=False)
    qs = conv_out[:, :H * K].view(L, H, K).float()
    ks = conv_out[:, H * K:2 * H * K].view(L, H, K).float()
    vs = conv_out[:, 2 * H * K:].view(L, HV, V)
    qref = qs * torch.rsqrt((qs * qs).sum(-1, keepdim=True) + 1e-6)
    kref = ks * torch.rsqrt((ks * ks).sum(-1, keepdim=True) + 1e-6)
    check("T3a post_conv q", q, qref.to(DT))
    check("T3b post_conv k", k, kref.to(DT))
    check("T3c post_conv v", v, vs)
    x = (a.float() + dt_bias.unsqueeze(0))
    sp = torch.where(x <= 20.0, F.softplus(x), x)
    gref = -torch.exp(A_log).unsqueeze(0) * sp
    check("T3d post_conv g", ggo, gref)
    check("T3e post_conv beta", beta, torch.sigmoid(b.float()))


# ---------------- T4: chunk_gated_delta_rule -------------------------------

def test_chunk():
    H, HV, K, V = 8, 24, 128, 128
    scale = K ** -0.5
    cases = [
        ("seq64", [64]), ("seq63", [63]), ("seq65", [65]), ("seq128", [128]),
        ("seq300", [300]), ("multi 200+150+300", [200, 150, 300]),
        ("chunk-mix 64+64", [64, 64]), ("mix 1+129", [1, 129]),
    ]
    for name, lens in cases:
        L = sum(lens)
        N = len(lens)
        ggen = torch.Generator(device=DEV).manual_seed(L + N)
        q = F.normalize(torch.randn(1, L, H, K, generator=ggen, device=DEV, dtype=DT), dim=-1)
        k = F.normalize(torch.randn(1, L, H, K, generator=ggen, device=DEV, dtype=DT), dim=-1)
        v = torch.randn(1, L, HV, V, generator=ggen, device=DEV, dtype=DT)
        A_log, dt_bias, av, bv = gating_params(HV, L, 77)
        x = av.float() + dt_bias.unsqueeze(0)
        g = -torch.exp(A_log).unsqueeze(0) * F.softplus(x).clamp(max=20.0)
        g = g.unsqueeze(0)                                    # [1,L,HV]
        beta = torch.sigmoid(bv.float()).unsqueeze(0)         # [1,L,HV]
        init = torch.randn(N, HV, V, K, generator=ggen, device=DEV, dtype=torch.float32) * 0.05
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEV)
        o, fs = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=init,
            output_final_state=True, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False)
        # naive reference per sequence
        o_parts, s_parts = [], []
        st = 0
        for i, ln in enumerate(lens):
            oi, si = ref_recurrent(
                q[0, st:st + ln], k[0, st:st + ln], v[0, st:st + ln],
                g[0, st:st + ln], beta[0, st:st + ln], init[i], scale)
            o_parts.append(oi)
            s_parts.append(si)
            st += ln
        check("T4o chunk out [%s]" % name,
              torch.cat(o_parts).unsqueeze(0), o)
        check("T4s chunk state [%s]" % name,
              torch.stack(s_parts), fs)


# ---------------- T5: fused_sigmoid_gating_delta_rule_update (decode) ------

def test_sigmoid_gating_decode():
    H, HV, K, V = 8, 24, 128, 128
    scale = K ** -0.5
    for B in (1, 8):
        T = B  # one token per sequence
        ggen = torch.Generator(device=DEV).manual_seed(500 + B)
        q = torch.randn(B, 1, H, K, generator=ggen, device=DEV, dtype=DT)
        k = torch.randn(B, 1, H, K, generator=ggen, device=DEV, dtype=DT)
        v = torch.randn(B, 1, HV, V, generator=ggen, device=DEV, dtype=DT)
        A_log, dt_bias, av, bv = gating_params(HV, T, 55)
        init = torch.zeros(B + 1, HV, V, K, device=DEV, dtype=torch.float32)
        init[1:] = torch.randn(B, HV, V, K, generator=ggen, device=DEV, dtype=torch.float32) * 0.05
        idx = torch.arange(1, B + 1, dtype=torch.int32, device=DEV)
        init_snap = init.clone()   # kernel mutates init in-place; ref must use S0
        # kernel: B dim = batch of seqs; pass q as [B,T,H,K] no cu_seqlens
        o, fs = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=av, b=bv, dt_bias=dt_bias, q=q, k=k, v=v,
            initial_state=init, inplace_final_state=True,
            ssm_state_indices=idx.unsqueeze(1), use_qk_l2norm_in_kernel=True)
        # naive with same internal gating math
        x = av.float() + dt_bias.unsqueeze(0)
        sp = torch.where(x <= 20.0, F.softplus(x), x)
        g = -torch.exp(A_log).unsqueeze(0) * sp
        beta = torch.sigmoid(bv.float())
        qn = q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        kn = k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        o_parts, s_parts = [], []
        for i in range(B):
            oi, si = ref_recurrent(qn[i], kn[i], v[i].float(), g[i:i + 1],
                                   beta[i:i + 1], init_snap[idx[i].long()], scale)
            o_parts.append(oi)
            s_parts.append(si)
        check("T5o sigmoid-decode out B=%d" % B,
              torch.stack(o_parts).float(), o)
        check("T5s sigmoid-decode state B=%d" % B,
              torch.stack(s_parts), fs[idx.long()])


# ---------------- T6: packed decode ----------------------------------------

def test_packed_decode():
    H, HV, K, V = 8, 24, 128, 128
    scale = K ** -0.5
    for B in (1, 8):
        ggen = torch.Generator(device=DEV).manual_seed(900 + B)
        qkv = torch.randn(B, 2 * H * K + HV * V, generator=ggen, device=DEV, dtype=DT)
        A_log, dt_bias, av, bv = gating_params(HV, B, 66)
        init = torch.zeros(B + 1, HV, V, K, device=DEV, dtype=torch.float32)
        init[1:] = torch.randn(B, HV, V, K, generator=ggen, device=DEV, dtype=torch.float32) * 0.05
        idx = torch.arange(1, B + 1, dtype=torch.int32, device=DEV)
        out = torch.zeros(B, 1, HV, V, device=DEV, dtype=DT)
        init_snap = init.clone()   # packed decode writes state in-place
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=qkv, a=av, b=bv, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=init, out=out,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        q = qkv[:, :H * K].view(B, 1, H, K)
        k = qkv[:, H * K:2 * H * K].view(B, 1, H, K)
        v = qkv[:, 2 * H * K:].view(B, 1, HV, V)
        x = av.float() + dt_bias.unsqueeze(0)
        g = -torch.exp(A_log).unsqueeze(0) * F.softplus(x).clamp(max=20.0)
        beta = torch.sigmoid(bv.float())
        qn = q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        kn = k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        o_parts, s_parts = [], []
        for i in range(B):
            oi, si = ref_recurrent(qn[i], kn[i], v[i].float(), g[i:i + 1],
                                   beta[i:i + 1], init_snap[idx[i].long()], scale)
            o_parts.append(oi)
            s_parts.append(si)
        check("T6o packed-decode out B=%d" % B,
              torch.stack(o_parts).float(), out.float())
        check("T6s packed-decode state B=%d" % B,
              torch.stack(s_parts), init[idx.long()])
        # invalid state index (0) must produce zeros and not touch state
        if B >= 2:
            idx2 = idx.clone()
            idx2[0] = 0
            out2 = torch.full((B, 1, HV, V), 7.0, device=DEV, dtype=DT)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=qkv, a=av, b=bv, A_log=A_log, dt_bias=dt_bias,
                scale=scale, initial_state=init, out=out2,
                ssm_state_indices=idx2, use_qk_l2norm_in_kernel=True)
            ok = torch.isnan(out2[0]).sum().item() == 0 and out2[0].abs().sum().item() == 0
            RESULTS.append(("T6x packed-decode null-index zeroed", bool(ok), 0, 0, 0))
            print("T6x packed-decode null-index zeroed               %s"
                  % ("PASS" if ok else "FAIL"), flush=True)
            if not ok:
                FAILS.append("T6x")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("torch %s, device %s" % (torch.__version__, torch.cuda.get_device_name(0)))
    test_conv1d_fn()
    test_conv1d_update()
    test_post_conv_prep()
    test_chunk()
    test_sigmoid_gating_decode()
    test_packed_decode()
    print("=" * 70)
    npass = sum(1 for r in RESULTS if r[1])
    print("TOTAL %d/%d PASS" % (npass, len(RESULTS)))
    if FAILS:
        print("FAILURES:", *FAILS, sep="\n  ")
        sys.exit(1)
    print("ALL GDN KERNELS MATCH NAIVE REFERENCE")
