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
HIDDEN = 8192
DTYPE = torch.bfloat16

DEVICE = 'cuda'
N_STREAMS = 4
BATCH = 1
SINKHORN_ITERS = 20
SINKHORN_EPS = 1e-6
QUANTILES = [0.5, 0.2, 0.8]
PROVIDERS = ['tilelang', 'triton']
SEQLENS = [1024, 2048, 4096, 8192, 16384]


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
        # cutile's kernel was tuned for bf16 mix — keep its native dtype.
        return lambda: cutile_aggregate(x_nC, h_pre.to(dtype).contiguous())
    if provider == 'triton':
        # Pass fp32 mix to match tilelang's hard-coded fp32 requirement →
        # apples-to-apples vs tilelang. Triton's wrapper accepts fp32 H_pre
        # (`dtype is torch.float16 or torch.float32` per the docstring).
        x_Cn = x_nC.transpose(-1, -2).contiguous()
        return lambda: triton_aggregate(x_Cn, h_pre.contiguous(), n, True)
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        mix = h_pre.unsqueeze(-1).contiguous()  # fp32, kernel-required
        return lambda: tl_aggregate(x_nC, mix)
    raise ValueError(provider)


def build_expand_combine_fwd(provider, s, b, n, C, dtype):
    torch.manual_seed(0)
    residual = torch.randn(s, b, n, C, device=DEVICE, dtype=dtype)
    h_post = torch.randn(s, b, n, device=DEVICE, dtype=torch.float32)
    f = torch.randn(s, b, C, device=DEVICE, dtype=dtype)
    h_res = torch.randn(s, b, n, n, device=DEVICE, dtype=torch.float32)
    if provider == 'cutile':
        # cutile native dtype is bf16 for the mix tensors.
        hp = h_post.to(dtype).contiguous()
        hr = h_res.to(dtype).contiguous()
        return lambda: cutile_expand_combine(hr, residual, hp, f, None)
    if provider == 'triton':
        # Pass fp32 mix to match tilelang (apples-to-apples). Triton's
        # docstring confirms H_post / H_res accept fp16 or fp32.
        x_Cn = residual.transpose(-1, -2).contiguous()
        hp = h_post.contiguous()
        hr = h_res.contiguous()
        # New triton signature: (f, bias, H_post, x, H_res, use_tf32, fuse_grad_x_acc)
        return lambda: triton_expand_combine(f, None, hp, x_Cn, hr, n=4, use_tf32=True, fuse_grad_x_acc=False)
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        hp = h_post.unsqueeze(-1).contiguous()  # fp32 (kernel-required)
        hr = h_res.contiguous()                 # fp32 (kernel-required)
        return lambda: tl_expand_combine(f, residual, hp, hr)
    raise ValueError(provider)


def build_projection_fwd(provider, s, b, n, C, dtype):
    """Projection with RMSNorm `norm_weight` (gamma) — only triton + tilelang.

    cutile's `fused_proj_rms` doesn't accept a learnable RMSNorm weight, so
    the apples-to-apples comparison is between the two impls that do.
    """
    torch.manual_seed(0)
    M = s * b
    K = n * C
    N = 2 * n + n * n  # 24 for n=4
    x = torch.randn(M, K, device=DEVICE, dtype=dtype)
    norm_weight = torch.randn(K, device=DEVICE, dtype=torch.float32)

    if provider == 'triton':
        # phi as fp32 to match tilelang's required fn dtype (apples-to-apples).
        # Triton's docstring lists fp16/fp32; with norm_weight provided the
        # kernel promotes phi to fp32 internally anyway.
        phi = torch.randn(N, K, device=DEVICE, dtype=torch.float32)
        return lambda: triton_projection(x, phi, norm_weight=norm_weight, use_tf32=True)
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        assert M % 32 == 0 and K % 256 == 0
        x_nC = x.view(M, n, C)
        fn = torch.randn(N, K, device=DEVICE, dtype=torch.float32)
        return lambda: tl_projection(x_nC, fn, norm_weight, 1e-6,
                                     fuse_grad_acc=False, n_splits=1)
    raise ValueError(f"projection benchmark only supports triton + tilelang (got {provider!r})")


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
        h = h_pre.to(dtype).contiguous().detach().requires_grad_(True)  # bf16 (cutile native)
        out = cutile_aggregate(x_in, h)
        return lambda: torch.autograd.grad(
            out, [x_in, h], grad_outputs=grad_out, retain_graph=True
        )
    if provider == 'triton':
        x_Cn = x_nC.transpose(-1, -2).contiguous().detach().requires_grad_(True)
        # fp32 mix to match tilelang's required dtype (apples-to-apples).
        h = h_pre.contiguous().detach().requires_grad_(True)
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
        # cutile native dtype: bf16 mix.
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
        # fp32 mix to match tilelang (apples-to-apples).
        x_Cn = residual.transpose(-1, -2).contiguous().detach().requires_grad_(True)
        hp = h_post.contiguous().detach().requires_grad_(True)
        hr = h_res.contiguous().detach().requires_grad_(True)
        ff = f.detach().requires_grad_(True)
        # New triton signature: (f, bias, H_post, x, H_res, use_tf32, fuse_grad_x_acc)
        out = triton_expand_combine(ff, None, hp, x_Cn, hr, n=4, use_tf32=True, fuse_grad_x_acc=False)
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
    """Projection bwd with RMSNorm `norm_weight` — only triton + tilelang.

    Aligned I/O across providers:
      inputs:  x bf16 (M*K elements), phi/fn fp32 (N, K), norm_weight fp32 (K,)
      grad_out: fp32 (M, N) — N=24 for n=4

    Triton's wrapper returns H padded to (M, 32) plus an auxiliary `ms` (M,). To
    match tilelang's single (M, N) output we slice H[:, :N] and drop grad_ms.
    """
    torch.manual_seed(0)
    M = s * b
    K = n * C
    N = 2 * n + n * n

    if provider == 'triton':
        x = torch.randn(M, K, device=DEVICE, dtype=dtype, requires_grad=True)
        phi = torch.randn(N, K, device=DEVICE, dtype=torch.float32, requires_grad=True)
        norm_weight = torch.randn(K, device=DEVICE, dtype=torch.float32, requires_grad=True)
        H, _ = triton_projection(x, phi, norm_weight=norm_weight, use_tf32=True)
        H_valid = H[:, :N]
        grad_out = torch.randn_like(H_valid)
        return lambda: torch.autograd.grad(
            H_valid, [x, phi, norm_weight],
            grad_outputs=grad_out,
            retain_graph=True,
        )
    if provider == 'tilelang':
        assert dtype == torch.bfloat16
        assert M % 32 == 0 and K % 256 == 0
        x_in = torch.randn(M, K, device=DEVICE, dtype=dtype, requires_grad=True)
        fn = torch.randn(N, K, device=DEVICE, dtype=torch.float32, requires_grad=True)
        nw = torch.randn(K, device=DEVICE, dtype=torch.float32, requires_grad=True)
        x_nC = x_in.view(M, n, C)
        out = tl_projection(x_nC, fn, nw, 1e-6, fuse_grad_acc=False, n_splits=1)
        grad_out = torch.randn_like(out)
        return lambda: torch.autograd.grad(
            out, [x_in, fn, nw], grad_outputs=grad_out, retain_graph=True
        )
    raise ValueError(f"projection benchmark only supports triton + tilelang (got {provider!r})")


# %%
# ===========================================================================
# perf_report benchmarks. One pair (fwd, bwd) per op.
# ===========================================================================
_LINE_NAMES = [p.capitalize() for p in PROVIDERS]
_STYLES = [('green', '-'), ('blue', '-')]

# Projection only compares triton + tilelang because cutile's `fused_proj_rms`
# doesn't accept a learnable RMSNorm weight (`norm_weight`).
_PROJ_PROVIDERS = ['tilelang', 'triton']
_PROJ_LINE_NAMES = [p.capitalize() for p in _PROJ_PROVIDERS]
_PROJ_STYLES = [('green', '-'), ('blue', '-')]


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


def _make_proj_bench(plot_name):
    return triton.testing.Benchmark(
        x_names=['seqlen'],
        x_vals=SEQLENS,
        x_log=True,
        line_arg='provider',
        line_vals=_PROJ_PROVIDERS,
        line_names=_PROJ_LINE_NAMES,
        styles=_PROJ_STYLES,
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
benchmark_sinkhorn_fwd.run(show_plots=True, return_df=True, print_data=True)

# # %%
# @triton.testing.perf_report(_make_bench(f'mhc-aggregate-fwd-C{HIDDEN}'))
# def benchmark_aggregate_fwd(seqlen, provider):
#     return _run_and_time(
#         build_aggregate_fwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
#     )

# # %%
# benchmark_aggregate_fwd.run(show_plots=True, return_df=True, print_data=True)

# # %%
# @triton.testing.perf_report(_make_bench(f'mhc-expand_combine-fwd-C{HIDDEN}'))
# def benchmark_expand_combine_fwd(seqlen, provider):
#     return _run_and_time(
#         build_expand_combine_fwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
#     )


# # %%
# benchmark_expand_combine_fwd.run(show_plots=True, return_df=True, print_data=True)

# # %%
# @triton.testing.perf_report(_make_proj_bench(f'mhc-projection-fwd-C{HIDDEN}-with-norm-weight'))
# def benchmark_projection_fwd(seqlen, provider):
#     return _run_and_time(
#         build_projection_fwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
#     )


# # %%
# benchmark_projection_fwd.run(show_plots=True, return_df=True, print_data=True)

# %%
@triton.testing.perf_report(_make_bench(f'mhc-sinkhorn-bwd-C{HIDDEN}'))
def benchmark_sinkhorn_bwd(seqlen, provider):
    return _run_and_time(
        build_sinkhorn_bwd(provider, seqlen, BATCH, N_STREAMS, SINKHORN_ITERS)
    )

# %%
benchmark_sinkhorn_bwd.run(show_plots=True, return_df=True, print_data=True)

# # %%
# @triton.testing.perf_report(_make_bench(f'mhc-aggregate-bwd-C{HIDDEN}'))
# def benchmark_aggregate_bwd(seqlen, provider):
#     return _run_and_time(
#         build_aggregate_bwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
#     )

# # %%
# benchmark_aggregate_bwd.run(show_plots=True, return_df=True, print_data=True)

# # %%
# @triton.testing.perf_report(_make_bench(f'mhc-expand_combine-bwd-C{HIDDEN}'))
# def benchmark_expand_combine_bwd(seqlen, provider):
#     return _run_and_time(
#         build_expand_combine_bwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
#     )

# # %%
# benchmark_expand_combine_bwd.run(show_plots=True, return_df=True, print_data=True)

# # %%
# @triton.testing.perf_report(_make_proj_bench(f'mhc-projection-bwd-C{HIDDEN}-with-norm-weight'))
# def benchmark_projection_bwd(seqlen, provider):
#     return _run_and_time(
#         build_projection_bwd(provider, seqlen, BATCH, N_STREAMS, HIDDEN, DTYPE)
#     )

# # %%
# benchmark_projection_bwd.run(show_plots=True, return_df=True, print_data=True)



