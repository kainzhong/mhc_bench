import argparse
import os
import time

import torch
from torch.cuda import nvtx
import torch.cuda.profiler as profiler

from triton_kernels.mhc_ops import (
    mhc_fused_sinkhorn,
    mhc_fused_scale,
    mhc_fused_aggregate,
    mhc_fused_expand_combine,
    mhc_fused_projection,
)

from cutile_kernels.cutile_kernels import (
    FusedSinkhornKnopp,
    FusedHAggregate,
    FusedHPostBDA,
    FusedProjRms,
)

from tilelang_kernels.modeling.mhc.ops.ops import (
    mhc_post as tl_post,
    mhc_pre_apply_mix as tl_pre_apply_mix,
    mhc_pre_norm_fn as tl_pre_norm_fn,
    mhc_pre_split_mixes as tl_pre_split_mixes,
    sinkhorn_normalize as tl_sinkhorn,
)

# Allocate random inputs + ones-like grads once on first call and reuse for
# all subsequent warmup / measured iterations — otherwise the repeated
# `torch.randn` / `torch.ones_like` launches dominate the nsys profile and
# obscure the mhc kernel times.
_CACHE: dict[str, torch.Tensor] = {}


def _get(tag: str, build):
    t = _CACHE.get(tag)
    if t is None:
        t = build()
        _CACHE[tag] = t
    return t


def run_sinkhorn_triton(B, T, n, dtype, device, iters):
    x = _get("sinkhorn_tr_x",
             lambda: torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=True))
    y = mhc_fused_sinkhorn(x, n, iters=iters)
    y_grad = _get("sinkhorn_tr_grad", lambda: torch.ones_like(y))
    y.backward(y_grad)

def run_sinkhorn_cutile(B, T, n, dtype, device, iters):
    x = _get("sinkhorn_cu_x",
             lambda: torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=True))
    y = FusedSinkhornKnopp.apply(x, iters)
    y_grad = _get("sinkhorn_cu_grad", lambda: torch.ones_like(y))
    y.backward(y_grad)

def run_sinkhorn_tilelang(B, T, n, dtype, device, iters):
    # Tilelang sinkhorn expects fp32 input.
    x = _get("sinkhorn_tl_x",
             lambda: torch.randn((B, T, n, n), device=device, dtype=torch.float32, requires_grad=True))
    y = tl_sinkhorn(x, repeat=iters)
    y_grad = _get("sinkhorn_tl_grad", lambda: torch.ones_like(y))
    y.backward(y_grad)

def run_projection_triton(B, T, n, C, dtype, device):
    nC = n * C
    x = _get("proj_tr_x",
             lambda: torch.randn(B * T, nC, device=device, requires_grad=True, dtype=dtype))
    phi = _get("proj_tr_phi",
               lambda: torch.randn(24, nC, dtype=torch.float32, requires_grad=True, device=device))
    H, ms = mhc_fused_projection(x, phi)
    H_grad = _get("proj_tr_H_grad", lambda: torch.ones_like(H))
    ms_grad = _get("proj_tr_ms_grad", lambda: torch.ones_like(ms))
    torch.autograd.backward([H, ms], [H_grad, ms_grad])

def run_projection_cutile(B, T, n, C, dtype, device):
    nC = n * C
    N = 2 * n + n * n
    x = _get("proj_cu_x",
             lambda: torch.randn(B * T, nC, device=device, requires_grad=True, dtype=dtype))
    phi = _get("proj_cu_phi",
               lambda: torch.randn(N, nC, dtype=dtype, requires_grad=True, device=device))
    Hs, r = FusedProjRms.apply(x, phi)
    Hs_grad = _get("proj_cu_Hs_grad", lambda: torch.ones_like(Hs))
    r_grad = _get("proj_cu_r_grad", lambda: torch.ones_like(r))
    torch.autograd.backward([Hs, r], [Hs_grad, r_grad])

def _build_proj_tl_fn_pad(device, rows, nC, OUT_N):
    fn_pad = torch.zeros(rows, nC, device=device, dtype=torch.float32)
    with torch.no_grad():
        fn_pad[:OUT_N].normal_()
    fn_pad.requires_grad_(True)
    return fn_pad

def run_projection_tilelang(B, T, n, C, dtype, device):
    # Tilelang's natural projection boundary is mhc_pre_norm_fn (matmul +
    # RMS norm fused). Residual must be bf16 with last two dims (n, C); fn
    # must be fp32 (OUT_N, n*C). The inner kernel reads 32 rows of fn even
    # though OUT_N = 2n + n*n = 24, so back it with (32, n*C) storage and
    # pass the (OUT_N, n*C) view. fuse_grad_acc=False matches the fairness
    # choice from kernel_comparison.md.
    nC = n * C
    OUT_N = 2 * n + n * n
    x = _get("proj_tl_x",
             lambda: torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True))
    fn_pad = _get("proj_tl_fn_pad",
                  lambda: _build_proj_tl_fn_pad(device, 32, nC, OUT_N))
    out = tl_pre_norm_fn(x, fn_pad[:OUT_N], None, 1e-6, fuse_grad_acc=False, n_splits=1)
    out_grad = _get("proj_tl_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def run_scale_triton(B, T, n, dtype, device):
    N = 2 * n + n * n
    H = _get("scale_tr_H",
             lambda: torch.randn((B * T, 32), device=device, dtype=dtype, requires_grad=True))
    alpha = _get("scale_tr_alpha",
                 lambda: torch.randn((3,), device=device, dtype=dtype, requires_grad=True))
    beta = _get("scale_tr_beta",
                lambda: torch.randn((1, N), device=device, dtype=dtype, requires_grad=True))
    ms = _get("scale_tr_ms",
              lambda: torch.rand((B * T), device=device, dtype=dtype, requires_grad=True))
    h_pre, h_post, h_res = mhc_fused_scale(H, alpha, beta, ms, n)
    hp_grad = _get("scale_tr_hp_grad", lambda: torch.ones_like(h_pre))
    ho_grad = _get("scale_tr_ho_grad", lambda: torch.ones_like(h_post))
    hr_grad = _get("scale_tr_hr_grad", lambda: torch.ones_like(h_res))
    torch.autograd.backward([h_pre, h_post, h_res], [hp_grad, ho_grad, hr_grad])

def run_scale_tilelang(B, T, n, dtype, device):
    # Tilelang's scale boundary is pre_split_mixes: affine + sigmoid + split,
    # consumes already-RMS-normed input. post_mult_value=2.0 / pre_eps=0.0
    # matches triton's hardcoded 2*sigmoid and bare sigmoid.
    OUT_N = 2 * n + n * n
    # pre_split_mixes consumes the fp32 RMS-normed output of pre_norm_fn.
    input_mixes = _get("scale_tl_input",
                       lambda: torch.randn(B, T, OUT_N, device=device, dtype=torch.float32, requires_grad=True))
    mhc_scale = _get("scale_tl_scale",
                     lambda: torch.randn(3, device=device, dtype=torch.float32, requires_grad=True))
    mhc_base = _get("scale_tl_base",
                    lambda: torch.randn(OUT_N, device=device, dtype=torch.float32, requires_grad=True))
    pre, post, comb = tl_pre_split_mixes(input_mixes, mhc_scale, mhc_base, n, 2.0, 0.0)
    pre_grad = _get("scale_tl_pre_grad", lambda: torch.ones_like(pre))
    post_grad = _get("scale_tl_post_grad", lambda: torch.ones_like(post))
    comb_grad = _get("scale_tl_comb_grad", lambda: torch.ones_like(comb))
    torch.autograd.backward([pre, post, comb], [pre_grad, post_grad, comb_grad])

def run_aggregate_triton(B, T, n, C, dtype, device):
    x = _get("agg_tr_x",
             lambda: torch.randn(B, T, C, n, dtype=dtype, requires_grad=True, device=device))
    H_pre = _get("agg_tr_H_pre",
                 lambda: torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device))
    out = mhc_fused_aggregate(x, H_pre, n)
    out_grad = _get("agg_tr_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def run_aggregate_cutile(B, T, n, C, dtype, device):
    x = _get("agg_cu_x",
             lambda: torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device))
    H_pre = _get("agg_cu_H_pre",
                 lambda: torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device))
    out = FusedHAggregate.apply(x, H_pre)
    out_grad = _get("agg_cu_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def run_aggregate_tilelang(B, T, n, C, dtype, device):
    # Tilelang mhc_pre_apply_mix: x bf16 (..., n, C), mix fp32 (..., n, 1).
    x = _get("agg_tl_x",
             lambda: torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True))
    mix = _get("agg_tl_mix",
               lambda: torch.randn(B, T, n, 1, dtype=torch.float32, device=device, requires_grad=True))
    out = tl_pre_apply_mix(x, mix)
    out_grad = _get("agg_tl_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def run_expand_combine_triton(B, T, n, C, dtype, device):
    x = _get("ec_tr_x",
             lambda: torch.randn(B, T, C, n, dtype=dtype, requires_grad=True, device=device))
    H_post = _get("ec_tr_H_post",
                  lambda: torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device))
    H_res = _get("ec_tr_H_res",
                 lambda: torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device))
    f = _get("ec_tr_f",
             lambda: torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device))
    out = mhc_fused_expand_combine(f, None, H_post, x, H_res, n, True)
    out_grad = _get("ec_tr_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def run_expand_combine_cutile(B, T, n, C, dtype, device):
    x = _get("ec_cu_x",
             lambda: torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device))
    H_post = _get("ec_cu_H_post",
                  lambda: torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device))
    H_res = _get("ec_cu_H_res",
                 lambda: torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device))
    f = _get("ec_cu_f",
             lambda: torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device))
    out = FusedHPostBDA.apply(H_res, x, H_post, f, None)
    out_grad = _get("ec_cu_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def run_expand_combine_tilelang(B, T, n, C, dtype, device):
    # Same autograd-wrapper path as triton/cutile (mhc_post.apply, matches
    # correctness.py). x / residual: bf16; post_mix / comb_mix: fp32.
    x = _get("ec_tl_x",
             lambda: torch.randn(B, T, C, dtype=torch.bfloat16, device=device, requires_grad=True))
    residual = _get("ec_tl_residual",
                    lambda: torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True))
    post_mix = _get("ec_tl_post_mix",
                    lambda: torch.randn(B, T, n, 1, dtype=torch.float32, device=device, requires_grad=True))
    comb_mix = _get("ec_tl_comb_mix",
                    lambda: torch.randn(B, T, n, n, dtype=torch.float32, device=device, requires_grad=True))
    out = tl_post(x, residual, post_mix, comb_mix)
    out_grad = _get("ec_tl_grad", lambda: torch.ones_like(out))
    out.backward(out_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["sinkhorn", "projection", "scale", "aggregate", "expand_combine", "all"], required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--C", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = args.B
    T = args.T
    C = args.C
    n = 4

    print(f"Running {args.operation} with B={B}, T={T}")

    # Per-framework runners for one op. cutile has no scale kernel, so that
    # cell is a no-op on the cutile side.
    def run_op_fw(op, fw):
        if op == "sinkhorn":
            {
                "triton":   run_sinkhorn_triton,
                "cutile":   run_sinkhorn_cutile,
                "tilelang": run_sinkhorn_tilelang,
            }[fw](B, T, n, dtype, device, args.sinkhorn_iters)
        elif op == "projection":
            {
                "triton":   run_projection_triton,
                "cutile":   run_projection_cutile,
                "tilelang": run_projection_tilelang,
            }[fw](B, T, n, C, dtype, device)
        elif op == "scale":
            if fw == "cutile":
                return  # cutile has no scale kernel (pytorch fallback upstream).
            {
                "triton":   run_scale_triton,
                "tilelang": run_scale_tilelang,
            }[fw](B, T, n, dtype, device)
        elif op == "aggregate":
            {
                "triton":   run_aggregate_triton,
                "cutile":   run_aggregate_cutile,
                "tilelang": run_aggregate_tilelang,
            }[fw](B, T, n, C, dtype, device)
        elif op == "expand_combine":
            {
                "triton":   run_expand_combine_triton,
                "cutile":   run_expand_combine_cutile,
                "tilelang": run_expand_combine_tilelang,
            }[fw](B, T, n, C, dtype, device)

    def run_all_for_fw(fw):
        if args.operation == "all":
            for op in ("sinkhorn", "projection", "scale", "aggregate", "expand_combine"):
                run_op_fw(op, fw)
        else:
            run_op_fw(args.operation, fw)

    # Profile each framework separately: warmup that framework, then
    # cudaProfilerStart/Stop around the measured iters, then move on. This
    # isolates L2/cache state between frameworks so each gets a clean warmup.
    # nsys must be launched with `--capture-range-end=repeat` so every
    # Start/Stop pair is captured into the same .nsys-rep.
    frameworks = ["triton", "cutile", "tilelang"]
    for fw in frameworks:
        # Warmup this framework.
        for _ in range(args.warmup):
            run_all_for_fw(fw)
        torch.cuda.synchronize()

        # Measured iters.
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push(f"{fw}_{args.operation}_B{B}_T{T}_C{C}")
        for _ in range(args.iters):
            run_all_for_fw(fw)
        torch.cuda.synchronize()
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
