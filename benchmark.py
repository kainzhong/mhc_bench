#!/usr/bin/env python3
"""Benchmark mHC fused kernels: cutile vs tilelang vs triton.

Forward and backward benchmarked separately. n (mHC streams) is fixed at 4;
batch is fixed at 1; seqlen sweeps. These kernels depend only on the product
s*b, so a 1-D sweep over seqlen is representative.

Builders live in mhc_lib.py (shared with profile_one.py).
"""
import argparse
import os
import sys

import torch
import triton
import triton.testing

from mhc_lib import (
    OPS,
    PROVIDERS,
    build_op,
    is_cutile_available,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hidden', type=int, default=1024,
                   help='Per-stream hidden dim C (total hidden = n*C). Default: 1024')
    p.add_argument('--op', choices=OPS + ['all'], default='all')
    p.add_argument('--pass', dest='which_pass',
                   choices=['fwd', 'bwd', 'both'], default='both')
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--save-path', default='./bench_plots')
    return p.parse_args()


_ARGS = _parse_args()
HIDDEN = _ARGS.hidden
DTYPE = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[_ARGS.dtype]
BATCH = 1
SEQLENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
QUANTILES = [0.5, 0.2, 0.8]

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
    fn()  # prime JIT / autotune
    torch.cuda.synchronize()
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=QUANTILES)
    return ms, max_ms, min_ms


# ---- forward ---------------------------------------------------------------
@triton.testing.perf_report(_make_bench(f'mhc-sinkhorn-fwd-C{HIDDEN}'))
def benchmark_sinkhorn_fwd(seqlen, provider):
    return _run_and_time(build_op('sinkhorn', provider, 'fwd', seqlen, BATCH, HIDDEN, DTYPE))


@triton.testing.perf_report(_make_bench(f'mhc-aggregate-fwd-C{HIDDEN}'))
def benchmark_aggregate_fwd(seqlen, provider):
    return _run_and_time(build_op('aggregate', provider, 'fwd', seqlen, BATCH, HIDDEN, DTYPE))


@triton.testing.perf_report(_make_bench(f'mhc-expand_combine-fwd-C{HIDDEN}'))
def benchmark_expand_combine_fwd(seqlen, provider):
    return _run_and_time(build_op('expand_combine', provider, 'fwd', seqlen, BATCH, HIDDEN, DTYPE))


@triton.testing.perf_report(_make_bench(f'mhc-projection-fwd-C{HIDDEN}'))
def benchmark_projection_fwd(seqlen, provider):
    return _run_and_time(build_op('projection', provider, 'fwd', seqlen, BATCH, HIDDEN, DTYPE))


# ---- backward --------------------------------------------------------------
@triton.testing.perf_report(_make_bench(f'mhc-sinkhorn-bwd-C{HIDDEN}'))
def benchmark_sinkhorn_bwd(seqlen, provider):
    return _run_and_time(build_op('sinkhorn', provider, 'bwd', seqlen, BATCH, HIDDEN, DTYPE))


@triton.testing.perf_report(_make_bench(f'mhc-aggregate-bwd-C{HIDDEN}'))
def benchmark_aggregate_bwd(seqlen, provider):
    return _run_and_time(build_op('aggregate', provider, 'bwd', seqlen, BATCH, HIDDEN, DTYPE))


@triton.testing.perf_report(_make_bench(f'mhc-expand_combine-bwd-C{HIDDEN}'))
def benchmark_expand_combine_bwd(seqlen, provider):
    return _run_and_time(build_op('expand_combine', provider, 'bwd', seqlen, BATCH, HIDDEN, DTYPE))


@triton.testing.perf_report(_make_bench(f'mhc-projection-bwd-C{HIDDEN}'))
def benchmark_projection_bwd(seqlen, provider):
    return _run_and_time(build_op('projection', provider, 'bwd', seqlen, BATCH, HIDDEN, DTYPE))


FWD_FUNCS = {
    'sinkhorn': benchmark_sinkhorn_fwd,
    'aggregate': benchmark_aggregate_fwd,
    'expand_combine': benchmark_expand_combine_fwd,
    'projection': benchmark_projection_fwd,
}
BWD_FUNCS = {
    'sinkhorn': benchmark_sinkhorn_bwd,
    'aggregate': benchmark_aggregate_bwd,
    'expand_combine': benchmark_expand_combine_bwd,
    'projection': benchmark_projection_bwd,
}


def main():
    if not torch.cuda.is_available():
        print('CUDA is required.', file=sys.stderr)
        sys.exit(1)
    if not is_cutile_available():
        print('WARNING: cuTile unavailable; cutile column will error.', file=sys.stderr)

    os.makedirs(_ARGS.save_path, exist_ok=True)

    print(f'Device : {torch.cuda.get_device_name()}')
    print(f'hidden : {HIDDEN} (per-stream C; total = {4 * HIDDEN})')
    print(f'dtype  : {_ARGS.dtype}')
    print(f'batch  : {BATCH} (fixed)')
    print(f'seqlens: {SEQLENS}')
    print(f'pass   : {_ARGS.which_pass}')
    print(f'save_to: {_ARGS.save_path}')
    print()

    selected = OPS if _ARGS.op == 'all' else [_ARGS.op]
    run_fwd = _ARGS.which_pass in ('fwd', 'both')
    run_bwd = _ARGS.which_pass in ('bwd', 'both')

    for op in selected:
        if run_fwd:
            print(f'=== {op} fwd ===')
            FWD_FUNCS[op].run(print_data=True, show_plots=False, save_path=_ARGS.save_path)
            print()
        if run_bwd:
            print(f'=== {op} bwd ===')
            BWD_FUNCS[op].run(print_data=True, show_plots=False, save_path=_ARGS.save_path)
            print()


if __name__ == '__main__':
    main()
