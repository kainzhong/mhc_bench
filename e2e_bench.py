# %%
#!/usr/bin/env python3
"""Benchmark mHC fused kernels: cutile vs tilelang vs triton.

Forward and backward are benchmarked separately. n (mHC streams) is fixed at 4;
batch is fixed at 1; seqlen varies (kernels depend only on s*b).

Ops benchmarked (each fwd + bwd):
  sinkhorn        - Sinkhorn-Knopp projection on n x n matrices
  aggregate       - (s,b,n,C) x (s,b,n) -> (s,b,C)
  expand_combine  - H_res @ residual + H_post * x
  projection      - (M,K) @ weight^T; each impl stops at its natural boundary

Per-impl backward call site (all via `torch.autograd.grad` on the public op):
  cutile    - `fused_*` wrappers
  tilelang  - `mhc_*` op wrappers
  triton    - `mhc_fused_*` (the exposed path)
"""
import argparse
import os
import sys

import torch
import triton
import triton.testing

from cutile_kernels.cutile_kernels import (
    fused_sinkhorn as cutile_sinkhorn,
    fused_h_aggregate as cutile_aggregate,
    fused_h_post_bda as cutile_expand_combine,
    fused_proj_rms as cutile_proj_rms,
)
from tilelang_kernels.modeling.mhc.ops.ops import (
    sinkhorn_normalize as tl_sinkhorn,
    mhc_pre_apply_mix as tl_aggregate,
    mhc_post as tl_expand_combine,
    mhc_pre_norm_fn as tl_projection,
)
from triton_kernels.mhc_ops import (
    mhc_fused_sinkhorn as triton_sinkhorn,
    mhc_fused_aggregate as triton_aggregate,
    mhc_fused_expand_combine as triton_expand_combine,
    mhc_fused_projection as triton_projection,
)

# %%
HIDDEN = 2048
DTYPE = torch.bfloat16

DEVICE = 'cuda'
N_STREAMS = 4
BATCH = 1
SINKHORN_ITERS = 20
SINKHORN_EPS = 1e-6
QUANTILES = [0.5, 0.2, 0.8]
PROVIDERS = ['cutile', 'tilelang', 'triton']
SEQLENS = [512, 1024, 2048, 4096, 8192, 16384]


# %%
# ===========================================================================
# Forward builders (each returns a zero-arg callable for do_bench)
# ===========================================================================
def build_sinkhorn_fwd(provider, s, b, n, iters):
    torch.manual_seed(0)
    h_res = torch.randn(s, b, n, n, device=DEVICE, dtype=torch.float32)
    if provider == 'cutile':
        return lambda: cutile_sinkhorn(h_res, iters)
    if provider == 'triton':
        return lambda: triton_sinkhorn(h_res, n=n, recompute_hist=True, iters=iters)
    if provider == 'tilelang':
        return lambda: tl_sinkhorn(h_res, repeat=iters, eps=SINKHORN_EPS)
    raise ValueError(provider)


def build_aggregate_fwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    x_nC = torch.randn(s, b, n, C, device=DEVICE, dtype=dtype)
    h_pre = torch.randn(s, b, n, device=DEVICE, dtype=torch.float32)
    if provider == 'cutile':
        return lambda: cutile_aggregate(x_nC, h_pre.to(dtype).contiguous())
    if provider == 'triton':
        x_Cn = x_nC.transpose(-1, -2).contiguous()
        return lambda: triton_aggregate(x_Cn, h_pre.to(dtype).contiguous(), n, True)
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        mix = h_pre.unsqueeze(-1).contiguous()
        return lambda: tl_aggregate(x_nC, mix)
    raise ValueError(provider)


def build_expand_combine_fwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    residual = torch.randn(s, b, n, C, device=DEVICE, dtype=dtype)
    h_post = torch.randn(s, b, n, device=DEVICE, dtype=torch.float32)
    f = torch.randn(s, b, C, device=DEVICE, dtype=dtype)
    h_res = torch.randn(s, b, n, n, device=DEVICE, dtype=torch.float32)
    if provider == 'cutile':
        hp = h_post.to(dtype).contiguous()
        hr = h_res.to(dtype).contiguous()
        return lambda: cutile_expand_combine(hr, residual, hp, f, None)
    if provider == 'triton':
        x_Cn = residual.transpose(-1, -2).contiguous()
        hp = h_post.to(dtype).contiguous()
        hr = h_res.to(dtype).contiguous()
        return lambda: triton_expand_combine(f, None, hp, x_Cn, hr, n, True)
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        hp = h_post.unsqueeze(-1).contiguous().to(torch.float32)
        hr = h_res.contiguous().to(torch.float32)
        return lambda: tl_expand_combine(f, residual, hp, hr)
    raise ValueError(provider)


def build_projection_fwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    M = s * b
    K = n * C
    N = 2 * n + n * n  # 24 for n=4
    x = torch.randn(M, K, device=DEVICE, dtype=dtype)

    if provider == 'cutile':
        weight = torch.randn(N, K, device=DEVICE, dtype=dtype)
        return lambda: cutile_proj_rms(x, weight)
    if provider == 'triton':
        phi = torch.randn(N, K, device=DEVICE, dtype=dtype)
        return lambda: triton_projection(x, phi, True)
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        assert M % 32 == 0 and K % 256 == 0
        x_nC = x.view(M, n, C)
        fn = torch.randn(N, K, device=DEVICE, dtype=torch.float32)
        return lambda: tl_projection(x_nC, fn, None, 1e-6,
                                     fuse_grad_acc=False, n_splits=1)
    raise ValueError(provider)


# %%
# ===========================================================================
# Backward builders — each runs fwd once (outside timing) to prep saved state,
# then returns a closure that invokes only the bwd kernel(s).
# ===========================================================================
def build_sinkhorn_bwd(provider, s, b, n, iters):
    torch.manual_seed(0)
    h_res = torch.randn(s, b, n, n, device=DEVICE, dtype=torch.float32)
    grad_out = torch.randn(s, b, n, n, device=DEVICE, dtype=torch.float32)

    if provider == 'cutile':
        inp = h_res.detach().clone().requires_grad_(True)
        out = cutile_sinkhorn(inp, iters, SINKHORN_EPS)
        return lambda: torch.autograd.grad(
            out, [inp], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'triton':
        inp = h_res.detach().clone().requires_grad_(True)
        out = triton_sinkhorn(inp, n=n, recompute_hist=True, iters=iters)
        return lambda: torch.autograd.grad(
            out, [inp], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'tilelang':
        inp = h_res.detach().clone().requires_grad_(True)
        out = tl_sinkhorn(inp, repeat=iters, eps=SINKHORN_EPS)
        return lambda: torch.autograd.grad(
            out, [inp], grad_outputs=grad_out, retain_graph=True
        )
    raise ValueError(provider)


def build_aggregate_bwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    x_nC = torch.randn(s, b, n, C, device=DEVICE, dtype=dtype)
    h_pre = torch.randn(s, b, n, device=DEVICE, dtype=torch.float32)
    grad_out = torch.randn(s, b, C, device=DEVICE, dtype=dtype)

    if provider == 'cutile':
        x_in = x_nC.detach().clone().requires_grad_(True)
        h = h_pre.to(dtype).contiguous().detach().requires_grad_(True)
        out = cutile_aggregate(x_in, h)
        return lambda: torch.autograd.grad(
            out, [x_in, h], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'triton':
        x_Cn = x_nC.transpose(-1, -2).contiguous().detach().requires_grad_(True)
        h = h_pre.to(dtype).contiguous().detach().requires_grad_(True)
        out = triton_aggregate(x_Cn, h, n, True)
        return lambda: torch.autograd.grad(
            out, [x_Cn, h], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        x_in = x_nC.detach().clone().requires_grad_(True)
        mix = h_pre.unsqueeze(-1).contiguous().detach().requires_grad_(True)
        out = tl_aggregate(x_in, mix)
        return lambda: torch.autograd.grad(
            out, [x_in, mix], grad_outputs=grad_out, retain_graph=True
        )
    raise ValueError(provider)


def build_expand_combine_bwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    residual = torch.randn(s, b, n, C, device=DEVICE, dtype=dtype)
    h_post = torch.randn(s, b, n, device=DEVICE, dtype=torch.float32)
    f = torch.randn(s, b, C, device=DEVICE, dtype=dtype)
    h_res = torch.randn(s, b, n, n, device=DEVICE, dtype=torch.float32)

    if provider == 'cutile':
        hr = h_res.to(dtype).contiguous().detach().requires_grad_(True)
        res_in = residual.detach().clone().requires_grad_(True)
        hp = h_post.to(dtype).contiguous().detach().requires_grad_(True)
        f_in = f.detach().clone().requires_grad_(True)
        out = cutile_expand_combine(hr, res_in, hp, f_in, None)
        grad_out = torch.randn_like(out)
        return lambda: torch.autograd.grad(
            out, [hr, res_in, hp, f_in], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'triton':
        x_Cn = residual.transpose(-1, -2).contiguous().detach().requires_grad_(True)
        hp = h_post.to(dtype).contiguous().detach().requires_grad_(True)
        hr = h_res.to(dtype).contiguous().detach().requires_grad_(True)
        ff = f.detach().requires_grad_(True)
        out = triton_expand_combine(ff, None, hp, x_Cn, hr, n, True)
        grad_out = torch.randn_like(out)
        return lambda: torch.autograd.grad(
            out, [ff, hp, x_Cn, hr], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        f_in = f.detach().clone().requires_grad_(True)
        res_in = residual.detach().clone().requires_grad_(True)
        hp = h_post.unsqueeze(-1).to(torch.float32).contiguous().detach().requires_grad_(True)
        hr = h_res.to(torch.float32).contiguous().detach().requires_grad_(True)
        out = tl_expand_combine(f_in, res_in, hp, hr)
        grad_out = torch.randn_like(out)
        return lambda: torch.autograd.grad(
            out, [f_in, res_in, hp, hr], grad_outputs=grad_out, retain_graph=True
        )
    raise ValueError(provider)


def build_projection_bwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    M = s * b
    K = n * C
    N = 2 * n + n * n

    x = torch.randn(M, K, device=DEVICE, dtype=dtype)

    if provider == 'cutile':
        weight = torch.randn(N, K, device=DEVICE, dtype=dtype)
        x_in = x.detach().clone().requires_grad_(True)
        w_in = weight.detach().clone().requires_grad_(True)
        proj, r = cutile_proj_rms(x_in, w_in, 1e-6)
        grad_proj = torch.randn_like(proj)
        grad_r = torch.randn_like(r)
        return lambda: torch.autograd.grad(
            [proj, r], [x_in, w_in],
            grad_outputs=[grad_proj, grad_r],
            retain_graph=True,
        )
    if provider == 'triton':
        phi = torch.randn(N, K, device=DEVICE, dtype=dtype)
        x_req = x.detach().clone().requires_grad_(True)
        phi_req = phi.detach().clone().requires_grad_(True)
        H, ms = triton_projection(x_req, phi_req, True)
        grad_H = torch.randn_like(H)
        grad_ms = torch.randn_like(ms)
        return lambda: torch.autograd.grad(
            [H, ms], [x_req, phi_req],
            grad_outputs=[grad_H, grad_ms],
            retain_graph=True,
        )
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        assert M % 32 == 0 and K % 256 == 0
        x_in = x.view(M, n, C).detach().clone().requires_grad_(True)
        fn = torch.randn(N, K, device=DEVICE, dtype=torch.float32).requires_grad_(True)
        out = tl_projection(x_in, fn, None, 1e-6, fuse_grad_acc=False, n_splits=1)
        grad_out = torch.randn_like(out)
        return lambda: torch.autograd.grad(
            out, [x_in, fn], grad_outputs=grad_out, retain_graph=True
        )
    raise ValueError(provider)


# %%
# ===========================================================================
# perf_report benchmarks. One pair (fwd, bwd) per op.
# ===========================================================================
_LINE_NAMES = [p.capitalize() for p in PROVIDERS]
_STYLES = [('red', '-'), ('green', '-'), ('blue', '-')]


def _make_bench(plot_name):
    return triton.testing.Benchmark(
        x_names=['seqlen'],
        x_vals=SEQLENS,
        x_log=True,
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=_LINE_NAMES,
        styles=_STYLES,
        ylabel='ms',
        plot_name=plot_name,
        args={},
    )


def _run_and_time(fn):
    fn()  # prime (JIT / autotune)
    torch.cuda.synchronize()
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    return ms, max_ms, min_ms


# %%
# ---- forward ---------------------------------------------------------------
@triton.testing.perf_report(_make_bench(f'mhc-sinkhorn-fwd-C{HIDDEN}'))
def benchmark_sinkhorn_fwd(seqlen, provider):
    return _run_and_time(
        build_sinkhorn_fwd(provider, seqlen, BATCH, N_STREAMS, SINKHORN_ITERS)
    )


# %%
benchmark_sinkhorn_fwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-aggregate-fwd-C{HIDDEN}'))
def benchmark_aggregate_fwd(seqlen, provider):
    return _run_and_time(
        build_aggregate_fwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
    )

# %%
benchmark_aggregate_fwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-expand_combine-fwd-C{HIDDEN}'))
def benchmark_expand_combine_fwd(seqlen, provider):
    return _run_and_time(
        build_expand_combine_fwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
    )


# %%
benchmark_expand_combine_fwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-projection-fwd-C{HIDDEN}'))
def benchmark_projection_fwd(seqlen, provider):
    return _run_and_time(
        build_projection_fwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
    )


# %%
benchmark_projection_fwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-sinkhorn-bwd-C{HIDDEN}'))
def benchmark_sinkhorn_bwd(seqlen, provider):
    return _run_and_time(
        build_sinkhorn_bwd(provider, seqlen, BATCH, N_STREAMS, SINKHORN_ITERS)
    )

# %%
benchmark_sinkhorn_bwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-aggregate-bwd-C{HIDDEN}'))
def benchmark_aggregate_bwd(seqlen, provider):
    return _run_and_time(
        build_aggregate_bwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
    )

# %%
benchmark_aggregate_bwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-expand_combine-bwd-C{HIDDEN}'))
def benchmark_expand_combine_bwd(seqlen, provider):
    return _run_and_time(
        build_expand_combine_bwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
    )

# %%
benchmark_expand_combine_bwd.run(show_plots=True, return_df=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-projection-bwd-C{HIDDEN}'))
def benchmark_projection_bwd(seqlen, provider):
    return _run_and_time(
        build_projection_bwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
    )

# %%
benchmark_projection_bwd.run(show_plots=True, return_df=True)

# %%



