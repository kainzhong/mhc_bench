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

def run_sinkhorn_triton(B, T, n, dtype, device, iters):
    x = torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=True)
    y = mhc_fused_sinkhorn(x, n, iters=iters)
    y.backward(torch.ones_like(y))

def run_sinkhorn_cutile(B, T, n, dtype, device, iters):
    x = torch.randn((B, T, n, n), device=device, dtype=dtype, requires_grad=True)
    y = FusedSinkhornKnopp.apply(x, iters)
    y.backward(torch.ones_like(y))

def run_sinkhorn_tilelang(B, T, n, dtype, device, iters):
    # Tilelang sinkhorn expects fp32 input.
    x = torch.randn((B, T, n, n), device=device, dtype=torch.float32, requires_grad=True)
    y = tl_sinkhorn(x, repeat=iters)
    y.backward(torch.ones_like(y))

def run_sinkhorn(B, T, n, dtype, device, iters):
    run_sinkhorn_cutile(B, T, n, dtype, device, iters)
    run_sinkhorn_triton(B, T, n, dtype, device, iters)
    run_sinkhorn_tilelang(B, T, n, dtype, device, iters)


def run_projection_triton(B, T, n, C, dtype, device):
    nC = n * C
    x = torch.randn(B * T, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(24, nC, dtype=dtype, requires_grad=True, device="cuda")
    H, ms = mhc_fused_projection(x, phi)
    torch.autograd.backward([H, ms], [torch.ones_like(H), torch.ones_like(ms)])

def run_projection_cutile(B, T, n, C, dtype, device):
    nC = n * C
    N = 2 * n + n * n
    x = torch.randn(B * T, nC, device="cuda", requires_grad=True, dtype=dtype)
    phi = torch.randn(N, nC, dtype=dtype, requires_grad=True, device="cuda")
    Hs, r = FusedProjRms.apply(x, phi)
    torch.autograd.backward([Hs, r], [torch.ones_like(Hs), torch.ones_like(r)])

def run_projection_tilelang(B, T, n, C, dtype, device):
    # Tilelang's natural projection boundary is mhc_pre_norm_fn (matmul +
    # RMS norm fused). Residual must be bf16 with last two dims (n, C); fn
    # must be fp32 (OUT_N, n*C). The inner kernel reads 32 rows of fn even
    # though OUT_N = 2n + n*n = 24, so back it with (32, n*C) storage and
    # pass the (OUT_N, n*C) view. fuse_grad_acc=False matches the fairness
    # choice from kernel_comparison.md.
    nC = n * C
    OUT_N = 2 * n + n * n
    x = torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True)
    fn_pad = torch.zeros(32, nC, device=device, dtype=torch.float32)
    with torch.no_grad():
        fn_pad[:OUT_N].normal_()
    fn_pad.requires_grad_(True)
    out = tl_pre_norm_fn(x, fn_pad[:OUT_N], None, 1e-6, fuse_grad_acc=False, n_splits=1)
    out.backward(torch.ones_like(out))

def run_projection(B, T, n, C, dtype, device):
    run_projection_cutile(B, T, n, C, dtype, device)
    run_projection_triton(B, T, n, C, dtype, device)
    run_projection_tilelang(B, T, n, C, dtype, device)


def run_scale_triton(B, T, n, dtype, device):
    N = 2 * n + n * n
    H = torch.randn((B * T, 32), device=device, dtype=dtype, requires_grad=True)
    alpha = torch.randn((3,), device=device, dtype=dtype, requires_grad=True)
    beta = torch.randn((1, N), device=device, dtype=dtype, requires_grad=True)
    ms = torch.rand((B * T), device=device, dtype=dtype, requires_grad=True)
    h_pre, h_post, h_res = mhc_fused_scale(H, alpha, beta, ms, n)
    torch.autograd.backward(
        [h_pre, h_post, h_res],
        [torch.ones_like(h_pre), torch.ones_like(h_post), torch.ones_like(h_res)],
    )

def run_scale_tilelang(B, T, n, dtype, device):
    # Tilelang's scale boundary is pre_split_mixes: affine + sigmoid + split,
    # consumes already-RMS-normed input. post_mult_value=2.0 / pre_eps=0.0
    # matches triton's hardcoded 2*sigmoid and bare sigmoid.
    OUT_N = 2 * n + n * n
    # pre_split_mixes consumes the fp32 RMS-normed output of pre_norm_fn.
    input_mixes = torch.randn(B, T, OUT_N, device=device, dtype=torch.float32, requires_grad=True)
    mhc_scale = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    mhc_base = torch.randn(OUT_N, device=device, dtype=torch.float32, requires_grad=True)
    pre, post, comb = tl_pre_split_mixes(input_mixes, mhc_scale, mhc_base, n, 2.0, 0.0)
    torch.autograd.backward(
        [pre, post, comb],
        [torch.ones_like(pre), torch.ones_like(post), torch.ones_like(comb)],
    )

def run_scale(B, T, n, dtype, device):
    run_scale_triton(B, T, n, dtype, device)
    run_scale_tilelang(B, T, n, dtype, device)


def run_aggregate_triton(B, T, n, C, dtype, device):
    x = torch.randn(B, T, C, n, dtype=dtype, requires_grad=True, device=device)
    H_pre = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    out = mhc_fused_aggregate(x, H_pre, n)
    out.backward(torch.ones_like(out))

def run_aggregate_cutile(B, T, n, C, dtype, device):
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_pre = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    out = FusedHAggregate.apply(x, H_pre)
    out.backward(torch.ones_like(out))

def run_aggregate_tilelang(B, T, n, C, dtype, device):
    # Tilelang mhc_pre_apply_mix: x bf16 (..., n, C), mix fp32 (..., n, 1).
    x = torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True)
    mix = torch.randn(B, T, n, 1, dtype=torch.float32, device=device, requires_grad=True)
    out = tl_pre_apply_mix(x, mix)
    out.backward(torch.ones_like(out))

def run_aggregate(B, T, n, C, dtype, device):
    run_aggregate_cutile(B, T, n, C, dtype, device)
    run_aggregate_triton(B, T, n, C, dtype, device)
    run_aggregate_tilelang(B, T, n, C, dtype, device)


def run_expand_combine_triton(B, T, n, C, dtype, device):
    x = torch.randn(B, T, C, n, dtype=dtype, requires_grad=True, device=device)
    H_post = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    H_res = torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device)
    f = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device)
    out = mhc_fused_expand_combine(f, None, H_post, x, H_res, n, True)
    out.backward(torch.ones_like(out))

def run_expand_combine_cutile(B, T, n, C, dtype, device):
    x = torch.randn(B, T, n, C, dtype=dtype, requires_grad=True, device=device)
    H_post = torch.randn(B, T, n, dtype=dtype, requires_grad=True, device=device)
    H_res = torch.randn(B, T, n, n, dtype=dtype, requires_grad=True, device=device)
    f = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device=device)
    out = FusedHPostBDA.apply(H_res, x, H_post, f, None)
    out.backward(torch.ones_like(out))

def run_expand_combine_tilelang(B, T, n, C, dtype, device):
    # Same autograd-wrapper path as triton/cutile (mhc_post.apply, matches
    # correctness.py). x / residual: bf16; post_mix / comb_mix: fp32.
    x = torch.randn(B, T, C, dtype=torch.bfloat16, device=device, requires_grad=True)
    residual = torch.randn(B, T, n, C, dtype=torch.bfloat16, device=device, requires_grad=True)
    post_mix = torch.randn(B, T, n, 1, dtype=torch.float32, device=device, requires_grad=True)
    comb_mix = torch.randn(B, T, n, n, dtype=torch.float32, device=device, requires_grad=True)
    out = tl_post(x, residual, post_mix, comb_mix)
    out.backward(torch.ones_like(out))

def run_expand_combine(B, T, n, C, dtype, device):
    run_expand_combine_cutile(B, T, n, C, dtype, device)
    run_expand_combine_triton(B, T, n, C, dtype, device)
    run_expand_combine_tilelang(B, T, n, C, dtype, device)

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

    def run_op(op):
        if op == "sinkhorn":
            run_sinkhorn(B, T, n, dtype, device, args.sinkhorn_iters)
        elif op == "projection":
            run_projection(B, T, n, C, dtype, device)
        elif op == "scale":
            run_scale(B, T, n, dtype, device)
        elif op == "aggregate":
            run_aggregate(B, T, n, C, dtype, device)
        elif op == "expand_combine":
            run_expand_combine(B, T, n, C, dtype, device)
        elif op == "all":
            run_sinkhorn(B, T, n, dtype, device, args.sinkhorn_iters)
            run_projection(B, T, n, C, dtype, device)
            run_scale(B, T, n, dtype, device)
            run_aggregate(B, T, n, C, dtype, device)
            run_expand_combine(B, T, n, C, dtype, device)

    # Warmup
    for _ in range(args.warmup):
        run_op(args.operation)
    torch.cuda.synchronize()

    # Start profiling AFTER warmup/autotuning
    torch.cuda.cudart().cudaProfilerStart()

    nvtx_label = f"{args.operation}_B{B}_T{T}_C{C}"

    # Profile iterations
    nvtx.range_push(nvtx_label)
    for _ in range(args.iters):
        run_op(args.operation)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # Stop profiling
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
