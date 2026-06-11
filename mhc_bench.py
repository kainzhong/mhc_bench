import argparse
import os
import time

import torch
from torch.cuda import nvtx
import torch.cuda.profiler as profiler

from transformer_engine.pytorch.triton.mhc import (
    mhc_fused_sinkhorn,
    mhc_fused_scale,
    mhc_fused_aggregate,
    mhc_fused_expand_combine,
    mhc_fused_projection,
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
# `torch.randn` / `torch.randn_like` launches dominate the nsys profile and
# obscure the mhc kernel times.
_CACHE: dict[str, torch.Tensor] = {}


def _get(tag: str, build):
    t = _CACHE.get(tag)
    if t is None:
        t = build()
        _CACHE[tag] = t
    return t


# L2 flusher. B200 has ~126 MB L2 per die; 256 MB writes through L2 twice
# over and evicts any residue from the previous op. Without this, the small
# fp32 mix/H tensors (256 KB–1 MB each) stay hot in L2 across iters and the
# kernels measure with warm caches instead of cold HBM reads.
_L2_FLUSHER: torch.Tensor | None = None


def _flush_l2(device):
    global _L2_FLUSHER
    if _L2_FLUSHER is None:
        _L2_FLUSHER = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_FLUSHER.zero_()


def run_sinkhorn_triton(B, T, n, dtype, device, iters):
    # Input is h_res from mhc_fused_scale, which is fp32 in the production pipeline
    # (scale's H comes from projection's fp32 output and out dtype follows H).
    # Use randn for the gradient seed — uniform-ones gradients drive tilelang's
    # bwd recompute into denormals on intermediate (dy - mean(dy))-style ops,
    # making the comparison unfair to tilelang. randn matches realistic loss
    # gradients flowing back from downstream layers.
    x = _get("sinkhorn_tr_x",
             lambda: torch.randn((B, T, n, n), device=device, dtype=torch.float32, requires_grad=True))
    y = mhc_fused_sinkhorn(x, n, iters=iters)
    y_grad = _get("sinkhorn_tr_grad", lambda: torch.randn_like(y))
    y.backward(y_grad)

def run_sinkhorn_tilelang(B, T, n, dtype, device, iters):
    # Tilelang sinkhorn expects fp32 input. See run_sinkhorn_triton for the
    # randn-gradient rationale.
    x = _get("sinkhorn_tl_x",
             lambda: torch.randn((B, T, n, n), device=device, dtype=torch.float32, requires_grad=True))
    y = tl_sinkhorn(x, repeat=iters)
    y_grad = _get("sinkhorn_tl_grad", lambda: torch.randn_like(y))
    y.backward(y_grad)

def run_projection_triton(B, T, n, C, dtype, device):
    nC = n * C
    x = _get("proj_tr_x",
             lambda: torch.randn(B * T, nC, device=device, requires_grad=True, dtype=dtype))
    phi = _get("proj_tr_phi",
               lambda: torch.randn(24, nC, dtype=torch.float32, requires_grad=True, device=device))
    H, ms = mhc_fused_projection(x, phi)
    H_grad = _get("proj_tr_H_grad", lambda: torch.randn_like(H))
    ms_grad = _get("proj_tr_ms_grad", lambda: torch.randn_like(ms))
    torch.autograd.backward([H, ms], [H_grad, ms_grad])

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
    out_grad = _get("proj_tl_grad", lambda: torch.randn_like(out))
    out.backward(out_grad)

def run_scale_triton(B, T, n, dtype, device):
    # H and ms come from mhc_fused_projection (both fp32). alpha/beta should be fp32
    # per the DeepSeek paper. Output h_pre/h_post/h_res follow H's dtype (fp32).
    N = 2 * n + n * n
    H = _get("scale_tr_H",
             lambda: torch.randn((B * T, 32), device=device, dtype=torch.float32, requires_grad=True))
    alpha = _get("scale_tr_alpha",
                 lambda: torch.randn((3,), device=device, dtype=torch.float32, requires_grad=True))
    beta = _get("scale_tr_beta",
                lambda: torch.randn((1, N), device=device, dtype=torch.float32, requires_grad=True))
    ms = _get("scale_tr_ms",
              lambda: torch.rand((B * T), device=device, dtype=torch.float32, requires_grad=True))
    h_pre, h_post, h_res = mhc_fused_scale(H, alpha, beta, ms, n)
    hp_grad = _get("scale_tr_hp_grad", lambda: torch.randn_like(h_pre))
    ho_grad = _get("scale_tr_ho_grad", lambda: torch.randn_like(h_post))
    hr_grad = _get("scale_tr_hr_grad", lambda: torch.randn_like(h_res))
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
    pre_grad = _get("scale_tl_pre_grad", lambda: torch.randn_like(pre))
    post_grad = _get("scale_tl_post_grad", lambda: torch.randn_like(post))
    comb_grad = _get("scale_tl_comb_grad", lambda: torch.randn_like(comb))
    torch.autograd.backward([pre, post, comb], [pre_grad, post_grad, comb_grad])

def run_aggregate_triton(B, T, n, C, dtype, device):
    # H_pre comes from mhc_fused_scale, which outputs fp32 in production. x stays bf16.
    x = _get("agg_tr_x",
             lambda: torch.randn(B, T, C, n, dtype=dtype, requires_grad=True, device=device))
    H_pre = _get("agg_tr_H_pre",
                 lambda: torch.randn(B, T, n, dtype=torch.float32, requires_grad=True, device=device))
    out = mhc_fused_aggregate(x, H_pre, n)
    out_grad = _get("agg_tr_grad", lambda: torch.randn_like(out))
    out.backward(out_grad)

def run_aggregate_tilelang(B, T, n, C, dtype, device):
    # Tilelang mhc_pre_apply_mix: x bf16 (..., n, C), mix fp32 (..., n, 1).
    x = _get("agg_tl_x",
             lambda: torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True))
    mix = _get("agg_tl_mix",
               lambda: torch.randn(B, T, n, 1, dtype=torch.float32, device=device, requires_grad=True))
    out = tl_pre_apply_mix(x, mix)
    out_grad = _get("agg_tl_grad", lambda: torch.randn_like(out))
    out.backward(out_grad)

def run_expand_combine_triton(B, T, n, C, dtype, device):
    # H_post comes from scale (fp32 in production), H_res from sinkhorn fed by scale's
    # h_res (also fp32). x and f are bf16 activations.
    x = _get("ec_tr_x",
             lambda: torch.randn(B, T, C, n, dtype=dtype, requires_grad=True, device=device))
    H_post = _get("ec_tr_H_post",
                  lambda: torch.randn(B, T, n, dtype=torch.float32, requires_grad=True, device=device))
    H_res = _get("ec_tr_H_res",
                 lambda: torch.randn(B, T, n, n, dtype=torch.float32, requires_grad=True, device=device))
    f = _get("ec_tr_f",
             lambda: torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device))
    out = mhc_fused_expand_combine(f, None, H_post, x, H_res, n, True)
    out_grad = _get("ec_tr_grad", lambda: torch.randn_like(out))
    out.backward(out_grad)

def run_expand_combine_tilelang(B, T, n, C, dtype, device):
    # Same autograd-wrapper path as triton (mhc_post.apply, matches
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
    out_grad = _get("ec_tl_grad", lambda: torch.randn_like(out))
    out.backward(out_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["sinkhorn", "projection", "scale", "aggregate", "expand_combine", "all"], required=True)
    parser.add_argument("--framework", choices=["triton", "tilelang", "all"], default="all")
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

    # Per-framework runners for one op. Each invocation flushes L2 first so the
    # kernel reads cold from HBM (warmup paths flush too — measurement and
    # warmup must run under matching cache conditions).
    def run_op_fw(op, fw):
        _flush_l2(device)
        if op == "sinkhorn":
            {
                "triton":   run_sinkhorn_triton,
                "tilelang": run_sinkhorn_tilelang,
            }[fw](B, T, n, dtype, device, args.sinkhorn_iters)
        elif op == "projection":
            {
                "triton":   run_projection_triton,
                "tilelang": run_projection_tilelang,
            }[fw](B, T, n, C, dtype, device)
        elif op == "scale":
            {
                "triton":   run_scale_triton,
                "tilelang": run_scale_tilelang,
            }[fw](B, T, n, dtype, device)
        elif op == "aggregate":
            {
                "triton":   run_aggregate_triton,
                "tilelang": run_aggregate_tilelang,
            }[fw](B, T, n, C, dtype, device)
        elif op == "expand_combine":
            {
                "triton":   run_expand_combine_triton,
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
    frameworks = ["triton", "tilelang"] if args.framework == "all" else [args.framework]
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
