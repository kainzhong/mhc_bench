#!/usr/bin/env python3
"""Correctness check: each impl (cutile / tilelang / triton) is compared
against a pure-PyTorch golden reference using the autograd path on both
sides. Uses the reference precision convention from the triton tests:

  - torch.backends.cuda.matmul.allow_tf32 = False
  - torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
  - reference functions cast to fp32 internally and back to input dtype

…so the reference accumulates in fp32 (matching the kernels) and quantises to
the input dtype on return (matching the kernels' output). At bf16 that makes
`atol = rtol = 2.5e-2` a sensible bar.

All three impls expose an `autograd.Function`-backed wrapper (triton:
`triton_kernels.mhc_ops`; tilelang: `tilelang_kernels.modeling.mhc.ops.ops`;
cutile: `cutile_kernels.cutile_kernels`) so the same
`(ref_out.sum() + fused_out.sum()).backward()` pattern works for all of them.

Run with:  NVTE_DISABLE_TRITON_AUTOTUNING=1 python correctness.py
"""
import sys

import torch

# Match triton kernels' internal precision: fp32 accumulator for bf16 matmul.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# Reference impls (match the conventions used in the triton test harness).
from triton_kernels.mhc_ref import (
    mhc_projection_ref,
    mhc_sinkhorn_ref,
    mhc_aggregate_ref,
    mhc_expand_combine_ref,
)

from cutile_kernels.cutile_kernels import (
    fused_sinkhorn as cutile_sinkhorn,
    fused_h_aggregate as cutile_aggregate,
    fused_h_post_bda as cutile_expand_combine,
    fused_proj_rms as cutile_proj_rms,
    is_cutile_available,
)
from tilelang_kernels.modeling.mhc.ops.ops import (
    sinkhorn_normalize as tl_sinkhorn,
    mhc_pre_apply_mix as tl_pre_apply_mix,
    mhc_post as tl_post,
    mhc_pre_norm_fn as tl_pre_norm_fn,
)
from triton_kernels.mhc_ops import (
    mhc_fused_sinkhorn as triton_sinkhorn,
    mhc_fused_aggregate as triton_aggregate,
    mhc_fused_expand_combine as triton_expand_combine,
    mhc_fused_projection as triton_projection,
)


DEVICE = 'cuda'
N = 4
SEQLEN = 256
BATCH = 1
HIDDEN = 512  # per-stream C
DTYPE = torch.bfloat16
SINKHORN_ITERS = 20
SINKHORN_EPS = 1e-6
RMS_EPS = 1e-6


def get_tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2.5e-2, rtol=2.5e-2)
    return dict(atol=5e-3, rtol=5e-3)


# ===========================================================================
# Reporting helper
# ===========================================================================
def _chk(label, actual, expected, tols):
    a = actual.detach().float() if actual is not None else None
    e = expected.detach().float() if expected is not None else None
    if a is None or e is None:
        print(f'  [SKIP] {label}')
        return True
    if a.shape != e.shape:
        print(f'  [FAIL] {label:28s}  SHAPE {tuple(a.shape)} vs {tuple(e.shape)}')
        return False
    max_abs = (a - e).abs().max().item()
    max_rel = ((a - e).abs() / e.abs().clamp_min(1e-8)).max().item()
    try:
        torch.testing.assert_close(a, e, **tols)
        print(f'  [OK  ] {label:28s}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}')
        return True
    except AssertionError as err:
        first = str(err).splitlines()[0] if str(err) else repr(err)
        print(f'  [FAIL] {label:28s}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}')
        print(f'         {first}')
        return False


# ===========================================================================
# Sinkhorn
#
# All three impls take shape (s, b, n, n). triton uses log-space Sinkhorn as
# the ref does, so triton should be tight. cutile/tilelang use softmax +
# iterate; with iters=20 they should be within bf16 tolerance.
# ===========================================================================
def check_sinkhorn():
    print(f'=== sinkhorn (dtype={DTYPE}) ===')
    torch.manual_seed(0)
    s, b, n = SEQLEN, BATCH, N
    tols = get_tols(DTYPE)

    x = torch.randn(s, b, n, n, device=DEVICE, dtype=DTYPE, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    x_tr = x.detach().clone().requires_grad_(True)
    x_cu = x.detach().clone().requires_grad_(True)
    # tilelang wrapper wants fp32 input; run at fp32 throughout.
    x_tl = x.detach().to(torch.float32).view(s * b, n, n).clone().requires_grad_(True)

    out_ref = mhc_sinkhorn_ref(x_ref, n=n, iterations=SINKHORN_ITERS)
    out_tr = triton_sinkhorn(x_tr, n=n, recompute_hist=True, iters=SINKHORN_ITERS)
    out_cu = cutile_sinkhorn(x_cu, SINKHORN_ITERS)
    out_tl = tl_sinkhorn(x_tl, repeat=SINKHORN_ITERS, eps=SINKHORN_EPS).view(s, b, n, n)

    ok = True
    ok &= _chk('fwd triton', out_tr, out_ref, tols)
    ok &= _chk('fwd cutile', out_cu, out_ref, tols)
    ok &= _chk('fwd tilelang', out_tl, out_ref, tols)

    # Explicit contiguous grads: `.sum().backward()` produces stride-0 broadcast
    # grads that the tilelang sinkhorn bwd kernel rejects.
    out_ref.backward(torch.ones_like(out_ref))
    out_tr.backward(torch.ones_like(out_tr))
    out_cu.backward(torch.ones_like(out_cu))
    out_tl.backward(torch.ones_like(out_tl))

    ok &= _chk('bwd triton grad_x', x_tr.grad, x_ref.grad, tols)
    ok &= _chk('bwd cutile grad_x', x_cu.grad, x_ref.grad, tols)
    ok &= _chk('bwd tilelang grad_x', x_tl.grad.view(s, b, n, n), x_ref.grad, tols)
    return ok


# ===========================================================================
# Aggregate  — ref input shape is triton-convention (s, b, C, n).
# cutile takes (s, b, n, C); tilelang wrapper takes (..., n, C) with a mix of
# shape (..., n, 1) in fp32.
# ===========================================================================
def check_aggregate():
    print(f'=== aggregate (dtype={DTYPE}) ===')
    torch.manual_seed(0)
    s, b, C, n = SEQLEN, BATCH, HIDDEN, N
    tols = get_tols(DTYPE)

    x_Cn = torch.randn(s, b, C, n, device=DEVICE, dtype=DTYPE, requires_grad=True)
    H_pre = torch.randn(s, b, n, device=DEVICE, dtype=DTYPE, requires_grad=True)

    x_ref = x_Cn.detach().clone().requires_grad_(True)
    H_pre_ref = H_pre.detach().clone().requires_grad_(True)

    x_tr = x_Cn.detach().clone().requires_grad_(True)
    H_pre_tr = H_pre.detach().clone().requires_grad_(True)

    x_cu = x_Cn.detach().transpose(-1, -2).contiguous().requires_grad_(True)
    H_pre_cu = H_pre.detach().clone().requires_grad_(True)

    # tilelang wrapper: x as (..., n, C) bf16, mix as (..., n, 1) fp32.
    x_tl = (
        x_Cn.detach().transpose(-1, -2).contiguous().to(torch.bfloat16).requires_grad_(True)
    )
    H_pre_tl = (
        H_pre.detach().unsqueeze(-1).to(torch.float32).contiguous().requires_grad_(True)
    )

    out_ref = mhc_aggregate_ref(x_ref, H_pre_ref, n)
    out_tr = triton_aggregate(x_tr, H_pre_tr, n, False)
    out_cu = cutile_aggregate(x_cu, H_pre_cu)
    out_tl = tl_pre_apply_mix(x_tl, H_pre_tl)

    ok = True
    ok &= _chk('fwd triton', out_tr, out_ref, tols)
    ok &= _chk('fwd cutile', out_cu, out_ref, tols)
    ok &= _chk('fwd tilelang', out_tl, out_ref, tols)

    out_ref.backward(torch.ones_like(out_ref))
    out_tr.backward(torch.ones_like(out_tr))
    out_cu.backward(torch.ones_like(out_cu))
    out_tl.backward(torch.ones_like(out_tl))

    ok &= _chk('bwd triton grad_x', x_tr.grad, x_ref.grad, tols)
    ok &= _chk('bwd triton grad_H_pre', H_pre_tr.grad, H_pre_ref.grad, tols)
    ok &= _chk(
        'bwd cutile grad_x',
        x_cu.grad.transpose(-1, -2).contiguous(),
        x_ref.grad,
        tols,
    )
    ok &= _chk('bwd cutile grad_H_pre', H_pre_cu.grad, H_pre_ref.grad, tols)
    ok &= _chk(
        'bwd tilelang grad_x',
        x_tl.grad.transpose(-1, -2).contiguous(),
        x_ref.grad,
        tols,
    )
    ok &= _chk(
        'bwd tilelang grad_H_pre',
        H_pre_tl.grad.squeeze(-1),
        H_pre_ref.grad,
        tols,
    )
    return ok


# ===========================================================================
# Expand-combine  — ref x layout (s, b, C, n); output (s, b, C, n).
# cutile takes residual (s, b, n, C) and H_res in transposed convention.
# tilelang wrapper (mhc_post) takes residual (s, b, n, C), post_layer_mix
# (s, b, n, 1) fp32, comb_res_mix (s, b, n, n) fp32, and f (s, b, C) bf16.
# ===========================================================================
def check_expand_combine():
    print(f'=== expand_combine (dtype={DTYPE}) ===')
    torch.manual_seed(0)
    s, b, C, n = SEQLEN, BATCH, HIDDEN, N
    tols = get_tols(DTYPE)

    f = torch.randn(s, b, C, device=DEVICE, dtype=DTYPE, requires_grad=True)
    H_post = torch.randn(s, b, n, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_Cn = torch.randn(s, b, C, n, device=DEVICE, dtype=DTYPE, requires_grad=True)
    H_res = torch.randn(s, b, n, n, device=DEVICE, dtype=DTYPE, requires_grad=True)

    def clone_leaf(t):
        return t.detach().clone().requires_grad_(True)

    # ref
    f_ref = clone_leaf(f)
    Hp_ref = clone_leaf(H_post)
    x_ref = clone_leaf(x_Cn)
    Hr_ref = clone_leaf(H_res)

    # triton (same layout)
    f_tr = clone_leaf(f)
    Hp_tr = clone_leaf(H_post)
    x_tr = clone_leaf(x_Cn)
    Hr_tr = clone_leaf(H_res)

    # cutile (x as (n, C); H_res transposed convention)
    f_cu = clone_leaf(f)
    Hp_cu = clone_leaf(H_post)
    res_cu = x_Cn.detach().transpose(-1, -2).contiguous().requires_grad_(True)
    Hr_cu = H_res.detach().transpose(-1, -2).contiguous().requires_grad_(True)

    # tilelang: x bf16 (s, b, C); residual bf16 (s, b, n, C); post (s,b,n,1) fp32;
    # comb_res (s, b, n, n) fp32.
    f_tl = f.detach().to(torch.bfloat16).contiguous().requires_grad_(True)
    res_tl = (
        x_Cn.detach().transpose(-1, -2).contiguous().to(torch.bfloat16).requires_grad_(True)
    )
    Hp_tl = (
        H_post.detach().unsqueeze(-1).to(torch.float32).contiguous().requires_grad_(True)
    )
    Hr_tl = H_res.detach().to(torch.float32).contiguous().requires_grad_(True)

    out_ref = mhc_expand_combine_ref(f_ref, None, Hp_ref, x_ref, Hr_ref, n)
    out_tr = triton_expand_combine(f_tr, None, Hp_tr, x_tr, Hr_tr, n, False)
    # cutile returns (s, b, n, C) — transpose back to (s, b, C, n).
    out_cu_nC = cutile_expand_combine(Hr_cu, res_cu, Hp_cu, f_cu, None)
    out_cu = out_cu_nC.transpose(-1, -2).contiguous()
    # tilelang returns (s, b, n, C) — transpose back.
    out_tl_nC = tl_post(f_tl, res_tl, Hp_tl, Hr_tl)
    out_tl = out_tl_nC.transpose(-1, -2).contiguous()

    ok = True
    ok &= _chk('fwd triton', out_tr, out_ref, tols)
    ok &= _chk('fwd cutile', out_cu, out_ref, tols)
    ok &= _chk('fwd tilelang', out_tl, out_ref, tols)

    out_ref.backward(torch.ones_like(out_ref))
    out_tr.backward(torch.ones_like(out_tr))
    out_cu_nC.backward(torch.ones_like(out_cu_nC))
    out_tl_nC.backward(torch.ones_like(out_tl_nC))

    ok &= _chk('bwd triton grad_f', f_tr.grad, f_ref.grad, tols)
    ok &= _chk('bwd triton grad_H_post', Hp_tr.grad, Hp_ref.grad, tols)
    ok &= _chk('bwd triton grad_x', x_tr.grad, x_ref.grad, tols)
    ok &= _chk('bwd triton grad_H_res', Hr_tr.grad, Hr_ref.grad, tols)

    ok &= _chk('bwd cutile grad_f', f_cu.grad, f_ref.grad, tols)
    ok &= _chk('bwd cutile grad_H_post', Hp_cu.grad, Hp_ref.grad, tols)
    ok &= _chk(
        'bwd cutile grad_x',
        res_cu.grad.transpose(-1, -2).contiguous(),
        x_ref.grad,
        tols,
    )
    ok &= _chk(
        'bwd cutile grad_H_res',
        Hr_cu.grad.transpose(-1, -2).contiguous(),
        Hr_ref.grad,
        tols,
    )

    ok &= _chk('bwd tilelang grad_f', f_tl.grad, f_ref.grad, tols)
    ok &= _chk(
        'bwd tilelang grad_H_post',
        Hp_tl.grad.squeeze(-1),
        Hp_ref.grad,
        tols,
    )
    ok &= _chk(
        'bwd tilelang grad_x',
        res_tl.grad.transpose(-1, -2).contiguous(),
        x_ref.grad,
        tols,
    )
    ok &= _chk('bwd tilelang grad_H_res', Hr_tl.grad, Hr_ref.grad, tols)
    return ok


# ===========================================================================
# Projection  — each impl has a different output boundary, so each gets a
# ref that matches its boundary.
#   - triton: returns (H[:N], ms)                           → compare both
#   - cutile: returns (proj, r) where r = 1/(sqrt(ms)+eps)  → compare both
#   - tilelang (mhc_pre_norm_fn): returns fully RMS-normalized H[:N]
# ===========================================================================
def check_projection():
    print(f'=== projection (dtype={DTYPE}) ===')
    torch.manual_seed(0)
    s, b, C, n = SEQLEN, BATCH, HIDDEN, N
    M = s * b
    K = n * C
    OUT_N = 2 * n + n * n  # 24 for n=4
    assert M % 32 == 0 and K % 256 == 0
    tols = get_tols(DTYPE)

    # ---- Triton: (H_padded, ms) boundary ------------------------------------
    x_tr = torch.randn(M, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    phi_tr = torch.randn(OUT_N, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_ref = x_tr.detach().clone().requires_grad_(True)
    phi_ref = phi_tr.detach().clone().requires_grad_(True)

    ref_Hs, ref_ms = mhc_projection_ref(x_ref, phi_ref)
    tr_H_padded, tr_ms = triton_projection(x_tr, phi_tr, False)
    tr_H = tr_H_padded[:, :OUT_N]

    ok = True
    ok &= _chk('fwd triton H', tr_H, ref_Hs, tols)
    ok &= _chk('fwd triton ms', tr_ms, ref_ms, tols)

    (ref_Hs.sum() + ref_ms.sum()).backward()
    (tr_H.sum() + tr_ms.sum()).backward()

    ok &= _chk('bwd triton grad_x', x_tr.grad, x_ref.grad, tols)
    ok &= _chk('bwd triton grad_phi', phi_tr.grad, phi_ref.grad, tols)

    # ---- Cutile: (proj, r) boundary -----------------------------------------
    x_cu = torch.randn(M, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    w_cu = torch.randn(OUT_N, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_curef = x_cu.detach().clone().requires_grad_(True)
    w_curef = w_cu.detach().clone().requires_grad_(True)

    def cutile_proj_rms_ref(x_, w_):
        dt = x_.dtype
        xf = x_.to(torch.float32)
        wf = w_.to(torch.float32)
        proj = xf @ wf.t()
        ms = (xf * xf).mean(dim=-1, keepdim=True)
        r = 1.0 / (torch.sqrt(ms) + RMS_EPS)
        return proj.to(dt), r.to(dt)

    ref_proj, ref_r = cutile_proj_rms_ref(x_curef, w_curef)
    cu_proj, cu_r = cutile_proj_rms(x_cu, w_cu, RMS_EPS)

    ok &= _chk('fwd cutile proj', cu_proj, ref_proj, tols)
    ok &= _chk('fwd cutile r', cu_r, ref_r, tols)

    (ref_proj.sum() + ref_r.sum()).backward()
    (cu_proj.sum() + cu_r.sum()).backward()

    ok &= _chk('bwd cutile grad_x', x_cu.grad, x_curef.grad, tols)
    ok &= _chk('bwd cutile grad_w', w_cu.grad, w_curef.grad, tols)

    # ---- Tilelang (mhc_pre_norm_fn): fully RMS-normalized output ------------
    # The inner matmul kernel reads 32 rows of `fn` (even though mhc_mult3=24),
    # so fn is allocated with 32 rows and the (OUT_N, K) view is passed to the
    # wrapper. Residual is shaped as (M, n, C) to match the (..., mhc_mult,
    # hidden_size) contract.
    x_tl = (
        torch.randn(M, n, C, device=DEVICE, dtype=torch.bfloat16).contiguous()
        .requires_grad_(True)
    )
    fn_pad = torch.zeros(32, K, device=DEVICE, dtype=torch.float32)
    fn_pad[:OUT_N].normal_()
    fn_pad = fn_pad.requires_grad_(True)
    x_tlref = x_tl.detach().clone().requires_grad_(True)
    fn_pad_ref = fn_pad.detach().clone().requires_grad_(True)

    def tilelang_proj_rms_ref(x_, fn_pad_):
        xf = x_.to(torch.float32).reshape(-1, K)
        fn = fn_pad_[:OUT_N]
        proj = xf @ fn.t()
        ms = (xf * xf).mean(dim=-1, keepdim=True)
        return proj * torch.rsqrt(ms + RMS_EPS)

    ref_out_tl = tilelang_proj_rms_ref(x_tlref, fn_pad_ref)
    # Slice the 32-row padded fn to (OUT_N, K) for the wrapper (kernel reads 32
    # rows out of the underlying storage anyway).
    tl_out = tl_pre_norm_fn(
        x_tl,
        fn_pad[:OUT_N],
        None,
        RMS_EPS,
        fuse_grad_acc=False,
        n_splits=1,
    )

    ok &= _chk('fwd tilelang', tl_out, ref_out_tl, tols)

    ref_out_tl.backward(torch.ones_like(ref_out_tl))
    tl_out.backward(torch.ones_like(tl_out))

    ok &= _chk('bwd tilelang grad_x', x_tl.grad, x_tlref.grad, tols)
    # The wrapper returns grad for the (OUT_N, K) view; compare only those rows.
    ok &= _chk(
        'bwd tilelang grad_fn',
        fn_pad.grad[:OUT_N],
        fn_pad_ref.grad[:OUT_N],
        tols,
    )
    return ok


def main():
    if not torch.cuda.is_available():
        print('CUDA is required.', file=sys.stderr)
        sys.exit(1)
    if not is_cutile_available():
        print('cuTile is not available — cutile columns will error.', file=sys.stderr)
        sys.exit(1)

    print(f'Device : {torch.cuda.get_device_name()}')
    print(f'n      : {N}')
    print(f'seqlen : {SEQLEN}')
    print(f'batch  : {BATCH}')
    print(f'hidden : {HIDDEN} (per-stream C)')
    print(f'dtype  : {DTYPE}')
    print()

    all_ok = True
    all_ok &= check_sinkhorn(); print()
    all_ok &= check_aggregate(); print()
    all_ok &= check_expand_combine(); print()
    all_ok &= check_projection(); print()

    print('RESULT:', 'ALL OK' if all_ok else 'FAILURES ABOVE')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
