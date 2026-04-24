# How to compare mHC kernels from `nsys cuda_gpu_kern_sum`

Covers all three implementations in this repo — **triton**, **tilelang**,
**cutile** — so the mapping stays consistent whether you profile any two or
all three together.

## Kernel-name → (framework, op, direction) mapping

Each framework may launch one or more kernels per logical mHC op. To compare
per-op cost, **sum the kernels in each row below per framework**, then divide
one framework's sum by another's for the ratio.

| Op | Dir | Triton kernel(s) | Tilelang kernel(s) | Cutile kernel(s) |
|---|---|---|---|---|
| **proj_scale** | fwd | `_mhc_projection_fwd_fused` + `_mhc_scale_fwd_fused` | `_mhc_pre_norm_fn_fwd_mul_kernel_kernel` + `_mhc_pre_norm_fn_fwd_norm_kernel_kernel` + `mhc_pre_split_mixes_fwd_kernel_kernel` | `_ct_proj_rms_fwd_kernel_…` + **scale: not implemented (left to pytorch elementwise)** |
| **proj_scale** | bwd | `_mhc_projection_bwd_fused` + `_mhc_scale_bwd_fused` | `_mhc_pre_norm_fn_bwd_mul_kernel_kernel` + `_mhc_pre_norm_fn_bwd_norm_kernel_kernel` + `mhc_pre_split_mixes_bwd_kernel_kernel` | `_ct_proj_rms_bwd_kernel_…` + **scale: not implemented (left to pytorch elementwise)** |
| **sinkhorn**   | fwd | `_mhc_sinkhorn_fwd_fused_recompute` | `mhc_sinkhorn_kernel_kernel` | `_ct_sinkhorn_fwd_kernel_…` |
| **sinkhorn**   | bwd | `_mhc_sinkhorn_bwd_fused_recompute` | `mhc_sinkhorn_backward_kernel_kernel` | `_ct_sinkhorn_bwd_kernel_…` |
| **aggregate**  | fwd | `_mhc_aggregate_fwd` | `_mhc_pre_apply_mix_fwd_kernel_kernel` | `_ct_h_agg_fwd_kernel_…` |
| **aggregate**  | bwd | `_mhc_aggregate_bwd` | `_mhc_pre_apply_mix_bwd_kernel_kernel` | `_ct_h_agg_bwd_kernel_…` |
| **post**       | fwd | `_mhc_expand_combine_fwd` (or `_with_bias` variant) | `_mhc_post_fwd_kernel_kernel` | `_ct_hpb_fwd_kernel_…` |
| **post**       | bwd | `_mhc_expand_combine_bwd` (or `_with_bias` variant) | `_mhc_post_bwd_kernel_kernel` | `_ct_hpb_bwd_kernel_…` |

Kernel-name suffix conventions:
- **Tilelang** kernels end in `_kernel_kernel` (outer JIT function + inner
  `prim_func` name).
- **Cutile** kernels end in a long `_Kt1_A…p16_…` template-instantiation tag
  — everything after `_ct_<op>_<dir>_kernel_` is the layout/precision/tiling
  fingerprint the cutile compiler emits. Treat `_ct_<op>_<dir>_kernel_` as
  the stable prefix and ignore the trailing tag.

### Per-op comparison is fine — rsqrt placement is noise

The three frameworks draw the boundary between "projection" and "scale"
slightly differently:

- **triton `projection_*_fused`** (matmul + mean_sq) ↔ **tilelang
  `pre_norm_fn_*_mul`** (matmul + sum_sq) ↔ **cutile `proj_rms_*`** (matmul
  + RMS applied inline). Triton/tilelang stop at the statistic; cutile folds
  the rsqrt in.
- **triton `scale_*_fused`** (rsqrt + affine + sigmoid + split) ↔ **tilelang
  `pre_norm_fn_*_norm`** (rsqrt) + **`pre_split_mixes_*`** (affine + sigmoid
  + split) ↔ **cutile (no kernel — pytorch elementwise)**.

The rsqrt / divide step is O(M·N) — a few hundred thousand ops for
hidden=4096, M=8192, compared to ~800M FMAs in the matmul. That's ~4000×
smaller, so whether rsqrt lives in the "projection" kernel or the "scale"
kernel is <1% of either side's total cost. **Comparing per op directly is
fine** — just remember the boundary difference exists if you see a tiny gap
you can't otherwise explain.

For `proj_scale` with cutile: `fused_proj_rms` stops at matmul + RMS; there
is no cutile scale kernel. Label that cell "not implemented (pytorch
elementwise fallback)" — don't silently credit the missing work.

## Why tilelang uses >1 kernel for things that look elementwise

Short version: **tilelang refuses atomics to stay deterministic**. Cross-SM
reductions can't merge into the output in-place — they write per-SM partials
to HBM, then a second kernel reduces them. `triton_vs_tilelang.md` §3
("atomics vs persistent blocks") covers this.

Concrete example — projection:

- **Triton `_mhc_projection_fwd_fused`** uses a 2D grid `(pid_m, pid_k)`, and
  each `pid_k` block `tl.atomic_add`s its partial sum into the shared output.
  Split-K parallelism + atomic merge, all in one kernel.
- **Cutile `_ct_proj_rms_fwd_kernel_…`** also fuses matmul + RMS + output in
  one kernel; it's free to use atomics (no determinism guarantee).
- **Tilelang `pre_norm_fn_fwd_mul`** writes `(n_splits, M, 1, 24)` partials
  to HBM; **`pre_norm_fn_fwd_norm`** reads them back, reduces across
  `n_splits`, applies rsqrt. The boundary was designed for split-K, but the
  tilelang backend doesn't support split-K yet (`n_splits = 1` is hard-coded
  in `ops/norm_fn.py`) — so the second kernel currently adds overhead with
  none of the parallelism it was meant to enable.

Atomic contention is low on Hopper/Blackwell for this workload (many
independent M-tiles; only a handful of K-splits merge per M-tile;
`sem="relaxed"` drops ordering). That's why "1 atomic kernel" (triton,
cutile) beats "2 deterministic kernels" (tilelang) in the current numbers.

Non-determinism in triton is gated by `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` —
if you need bitwise-reproducible training, you can't use atomics and
tilelang's two-kernel structure is the price of determinism.
