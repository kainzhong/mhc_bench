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

For tilelang we wrap the JIT kernels in `torch.autograd.Function`s so the
same `(ref_out.sum() + fused_out.sum()).backward()` pattern works.

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
from tilelang_kernels.sinkhorn_kernel import (
    _mhc_sinkhorn_fwd as tl_sinkhorn_fwd_jit,
    _mhc_sinkhorn_bwd as tl_sinkhorn_bwd_jit,
)
from tilelang_kernels.pre_apply_mix_kernel import (
    _mhc_pre_apply_mix_fwd as tl_pre_apply_mix_fwd_jit,
    _mhc_pre_apply_mix_bwd as tl_pre_apply_mix_bwd_jit,
)
from tilelang_kernels.post_kernel import mhc_post_fwd as tl_post_fwd
from tilelang_kernels.post_kernel import mhc_post_bwd as tl_post_bwd
from tilelang_kernels.norm_fn_kernel import (
    _mhc_pre_norm_fn_fwd_mul as tl_norm_fn_mul_jit,
    _mhc_pre_norm_fn_fwd_norm as tl_norm_fn_norm_jit,
    _mhc_pre_norm_fn_bwd_mul as tl_norm_fn_bwd_mul_jit,
    _mhc_pre_norm_fn_bwd_norm as tl_norm_fn_bwd_norm_jit,
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


def _pick_token_block(num_tokens, preferred=32):
    for k in range(min(preferred, num_tokens), 0, -1):
        if num_tokens % k == 0:
            return k
    return 1


# ===========================================================================
# Tilelang autograd.Function wrappers — tilelang ships raw JIT kernels only,
# so we wrap them here to enable the same autograd-based test pattern.
# ===========================================================================
class TLSinkhorn(torch.autograd.Function):
    """tilelang sinkhorn; shape (num_tokens, n, n), kernel fp32."""

    @staticmethod
    def forward(ctx, h_res, n, iters, eps):
        in_dtype = h_res.dtype
        h_fp = h_res.to(torch.float32).contiguous()
        num_tokens = h_fp.shape[0]
        tb = _pick_token_block(num_tokens)
        fwd_k = tl_sinkhorn_fwd_jit(n, tb, iters, eps)
        out = torch.empty_like(h_fp)
        fwd_k(h_fp, out)
        ctx.save_for_backward(h_fp)
        ctx.n = n
        ctx.tb = tb
        ctx.iters = iters
        ctx.eps = eps
        ctx.in_dtype = in_dtype
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, go):
        (h_fp,) = ctx.saved_tensors
        bwd_k = tl_sinkhorn_bwd_jit(ctx.n, ctx.tb, ctx.iters, ctx.eps)
        gi = torch.empty_like(h_fp)
        go_fp = go.to(torch.float32).contiguous()
        bwd_k(go_fp, h_fp, gi)
        return gi.to(ctx.in_dtype), None, None, None


class TLAggregate(torch.autograd.Function):
    """tilelang pre_apply_mix; x: (sb,n,C) bf16, mix: (sb,n) fp32 -> (sb,C) bf16."""

    @staticmethod
    def forward(ctx, x, mix, n, C):
        ctx.in_dtype = x.dtype
        ctx.mix_dtype = mix.dtype
        x_bf = x.to(torch.bfloat16).contiguous()
        mix_fp = mix.to(torch.float32).contiguous()
        fwd_k = tl_pre_apply_mix_fwd_jit(n, C)
        sb = x_bf.shape[0]
        out = torch.empty(sb, C, device=x_bf.device, dtype=torch.bfloat16)
        fwd_k(x_bf, mix_fp, out)
        ctx.save_for_backward(x_bf, mix_fp)
        ctx.n = n
        ctx.C = C
        return out.to(ctx.in_dtype)

    @staticmethod
    def backward(ctx, go):
        x_bf, mix_fp = ctx.saved_tensors
        go_bf = go.to(torch.bfloat16).contiguous()
        bwd_k = tl_pre_apply_mix_bwd_jit(ctx.n, ctx.C)
        x_grad_bf = torch.zeros_like(x_bf)
        mix_grad_fp = bwd_k(go_bf, x_bf, mix_fp, x_grad_bf)
        return (
            x_grad_bf.to(ctx.in_dtype),
            mix_grad_fp.to(ctx.mix_dtype),
            None,
            None,
        )


class TLExpandCombine(torch.autograd.Function):
    """tilelang post kernel: f, residual (n,C), post_4d (s,b,n,1), comb_res (s,b,n,n)."""

    @staticmethod
    def forward(ctx, f, residual, post_layer_mix, comb_res_mix):
        ctx.in_dtype = f.dtype
        f_bf = f.to(torch.bfloat16).contiguous()
        res_bf = residual.to(torch.bfloat16).contiguous()
        pm_fp = post_layer_mix.to(torch.float32).contiguous()
        cr_fp = comb_res_mix.to(torch.float32).contiguous()
        out = tl_post_fwd(f_bf, res_bf, pm_fp, cr_fp)
        ctx.save_for_backward(f_bf, res_bf, pm_fp, cr_fp)
        return out.to(ctx.in_dtype)

    @staticmethod
    def backward(ctx, go):
        f_bf, res_bf, pm_fp, cr_fp = ctx.saved_tensors
        go_bf = go.to(torch.bfloat16).contiguous()
        d_x, d_res, d_pm, d_cr = tl_post_bwd(
            f_bf, res_bf, pm_fp, cr_fp, go_bf, fuse_grad_acc=False
        )
        return (
            d_x.to(ctx.in_dtype),
            d_res.to(ctx.in_dtype),
            d_pm.to(ctx.in_dtype),
            d_cr.to(ctx.in_dtype),
        )


class TLProjection(torch.autograd.Function):
    """tilelang norm_fn (projection + inline RMS). fn is padded to 32 rows
    because the kernel reads rows 0..31 even though mhc_mult3 < 32."""

    @staticmethod
    def forward(ctx, x, fn_padded, out_n, K):
        ctx.in_dtype = x.dtype
        x_bf = x.to(torch.bfloat16).contiguous()
        fn_pad_fp = fn_padded.to(torch.float32).contiguous()
        fn = fn_pad_fp[:out_n]

        M = x_bf.shape[0]
        mhc3, n_rms, rms_sz, n_splits = out_n, 1, K, 1
        mul_k = tl_norm_fn_mul_jit(mhc3, n_rms, rms_sz)
        norm_k = tl_norm_fn_norm_jit(mhc3, n_rms, rms_sz, RMS_EPS, n_splits)

        out_mul_split = torch.empty(
            n_splits, M, n_rms, mhc3, device=x_bf.device, dtype=torch.float32
        )
        sqrsum_split = torch.empty(
            n_splits, M, n_rms, device=x_bf.device, dtype=torch.float32
        )
        out_mul = torch.empty(M, n_rms, mhc3, device=x_bf.device, dtype=torch.float32)
        sqrsum = torch.empty(M, n_rms, device=x_bf.device, dtype=torch.float32)
        out = torch.empty(M, mhc3, device=x_bf.device, dtype=torch.float32)

        mul_k(x_bf, fn, out_mul_split[0], sqrsum_split[0])
        norm_k(out_mul_split, sqrsum_split, out_mul, sqrsum, out)

        ctx.save_for_backward(x_bf, fn_pad_fp, out_mul, sqrsum)
        ctx.out_n = out_n
        ctx.K = K
        return out.to(ctx.in_dtype)

    @staticmethod
    def backward(ctx, go):
        x_bf, fn_pad_fp, out_mul, sqrsum = ctx.saved_tensors
        out_n, K = ctx.out_n, ctx.K
        M = x_bf.shape[0]
        mhc3, n_rms, rms_sz = out_n, 1, K

        bwd_norm_k = tl_norm_fn_bwd_norm_jit(mhc3, n_rms, rms_sz, RMS_EPS)
        bwd_mul_k = tl_norm_fn_bwd_mul_jit(mhc3, n_rms, rms_sz)

        go_fp = go.to(torch.float32).contiguous()
        out_mul_grad = torch.empty(
            M, n_rms, mhc3, device=x_bf.device, dtype=torch.float32
        )
        sqrsum_grad = torch.empty(M, n_rms, device=x_bf.device, dtype=torch.float32)
        x_grad_bf = torch.zeros(M, K, device=x_bf.device, dtype=torch.bfloat16)
        fn_grad_pad = torch.zeros(32, K, device=x_bf.device, dtype=torch.float32)
        fn_grad = fn_grad_pad[:out_n]

        bwd_norm_k(go_fp, out_mul, sqrsum, out_mul_grad, sqrsum_grad)
        bwd_mul_k(out_mul_grad, sqrsum_grad, x_bf, fn_pad_fp[:out_n], x_grad_bf, fn_grad)

        # Build gradient w.r.t. the padded fn (only first out_n rows are used).
        fn_padded_grad = torch.zeros_like(fn_pad_fp)
        fn_padded_grad[:out_n] = fn_grad
        return x_grad_bf.to(ctx.in_dtype), fn_padded_grad, None, None


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
    x_tl = x.detach().view(s * b, n, n).clone().requires_grad_(True)

    out_ref = mhc_sinkhorn_ref(x_ref, n=n, iterations=SINKHORN_ITERS)
    out_tr = triton_sinkhorn(x_tr, n=n, recompute_hist=True, iters=SINKHORN_ITERS)
    out_cu = cutile_sinkhorn(x_cu, SINKHORN_ITERS)
    out_tl = TLSinkhorn.apply(x_tl, n, SINKHORN_ITERS, SINKHORN_EPS).view(s, b, n, n)

    ok = True
    ok &= _chk('fwd triton', out_tr, out_ref, tols)
    ok &= _chk('fwd cutile', out_cu, out_ref, tols)
    ok &= _chk('fwd tilelang', out_tl, out_ref, tols)

    out_ref.sum().backward()
    out_tr.sum().backward()
    out_cu.sum().backward()
    out_tl.sum().backward()

    ok &= _chk('bwd triton grad_x', x_tr.grad, x_ref.grad, tols)
    ok &= _chk('bwd cutile grad_x', x_cu.grad, x_ref.grad, tols)
    ok &= _chk('bwd tilelang grad_x', x_tl.grad.view(s, b, n, n), x_ref.grad, tols)
    return ok


# ===========================================================================
# Aggregate  — ref input shape is triton-convention (s, b, C, n).
# cutile takes (s, b, n, C); tilelang takes flat (sb, n, C) + fp32 mix.
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

    x_tl = x_Cn.detach().transpose(-1, -2).contiguous().view(s * b, n, C).requires_grad_(True)
    H_pre_tl = H_pre.detach().view(s * b, n).clone().requires_grad_(True)

    out_ref = mhc_aggregate_ref(x_ref, H_pre_ref, n)
    out_tr = triton_aggregate(x_tr, H_pre_tr, n, False)
    out_cu = cutile_aggregate(x_cu, H_pre_cu)
    out_tl = TLAggregate.apply(x_tl, H_pre_tl, n, C).view(s, b, C)

    ok = True
    ok &= _chk('fwd triton', out_tr, out_ref, tols)
    ok &= _chk('fwd cutile', out_cu, out_ref, tols)
    ok &= _chk('fwd tilelang', out_tl, out_ref, tols)

    out_ref.sum().backward()
    out_tr.sum().backward()
    out_cu.sum().backward()
    out_tl.sum().backward()

    ok &= _chk('bwd triton grad_x', x_tr.grad, x_ref.grad, tols)
    ok &= _chk('bwd triton grad_H_pre', H_pre_tr.grad, H_pre_ref.grad, tols)
    ok &= _chk(
        'bwd cutile grad_x',
        x_cu.grad.transpose(-1, -2).contiguous(),
        x_ref.grad,
        tols,
    )
    ok &= _chk('bwd cutile grad_H_pre', H_pre_cu.grad, H_pre_ref.grad, tols)
    tl_gx = x_tl.grad.view(s, b, n, C).transpose(-1, -2).contiguous()
    ok &= _chk('bwd tilelang grad_x', tl_gx, x_ref.grad, tols)
    ok &= _chk(
        'bwd tilelang grad_H_pre',
        H_pre_tl.grad.view(s, b, n),
        H_pre_ref.grad,
        tols,
    )
    return ok


# ===========================================================================
# Expand-combine  — ref x layout (s, b, C, n); output (s, b, C, n).
# cutile takes residual (s, b, n, C) and H_res in transposed convention.
# tilelang takes residual (s, b, n, C) and post as (s, b, n, 1).
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

    # tilelang (x as (n, C); H_post as (n, 1))
    f_tl = clone_leaf(f)
    Hp_tl_4d = H_post.detach().unsqueeze(-1).contiguous().requires_grad_(True)
    res_tl = x_Cn.detach().transpose(-1, -2).contiguous().requires_grad_(True)
    Hr_tl = clone_leaf(H_res)

    out_ref = mhc_expand_combine_ref(f_ref, None, Hp_ref, x_ref, Hr_ref, n)
    out_tr = triton_expand_combine(f_tr, None, Hp_tr, x_tr, Hr_tr, n, False)
    # cutile returns (s, b, n, C) — transpose back to (s, b, C, n).
    out_cu_nC = cutile_expand_combine(Hr_cu, res_cu, Hp_cu, f_cu, None)
    out_cu = out_cu_nC.transpose(-1, -2).contiguous()
    # tilelang returns (s, b, n, C) — transpose back.
    out_tl_nC = TLExpandCombine.apply(f_tl, res_tl, Hp_tl_4d, Hr_tl)
    out_tl = out_tl_nC.transpose(-1, -2).contiguous()

    ok = True
    ok &= _chk('fwd triton', out_tr, out_ref, tols)
    ok &= _chk('fwd cutile', out_cu, out_ref, tols)
    ok &= _chk('fwd tilelang', out_tl, out_ref, tols)

    out_ref.sum().backward()
    out_tr.sum().backward()
    out_cu_nC.sum().backward()
    out_tl_nC.sum().backward()

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
        Hp_tl_4d.grad.squeeze(-1),
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
#   - tilelang: returns fully RMS-normalized H[:N]          → compare that
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

    # ---- Tilelang: fully RMS-normalized output boundary ---------------------
    # fn is stored padded to 32 rows so the kernel's out-of-bounds row reads
    # hit valid memory.
    x_tl = torch.randn(M, K, device=DEVICE, dtype=DTYPE, requires_grad=True)
    fn_pad = torch.zeros(32, K, device=DEVICE, dtype=torch.float32)
    fn_pad[:OUT_N].normal_()
    fn_pad = fn_pad.requires_grad_(True)
    x_tlref = x_tl.detach().clone().requires_grad_(True)
    fn_pad_ref = fn_pad.detach().clone().requires_grad_(True)

    def tilelang_proj_rms_ref(x_, fn_pad_):
        xf = x_.to(torch.float32)
        fn = fn_pad_[:OUT_N]
        proj = xf @ fn.t()
        ms = (xf * xf).mean(dim=-1, keepdim=True)
        return proj * torch.rsqrt(ms + RMS_EPS)

    ref_out_tl = tilelang_proj_rms_ref(x_tlref, fn_pad_ref)
    tl_out = TLProjection.apply(x_tl, fn_pad, OUT_N, K)

    ok &= _chk('fwd tilelang', tl_out, ref_out_tl, tols)

    ref_out_tl.sum().backward()
    tl_out.sum().backward()

    ok &= _chk('bwd tilelang grad_x', x_tl.grad, x_tlref.grad, tols)
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
