# Triton vs. Tilelang mHC kernels — functional and optimization comparison

## Overview

The triton and tilelang kernel sets in this repo implement the **same five core
operations** of the mHC (manifold Hyper-Connection) block. They do it with
different algorithms, different fusion boundaries, and different optimization
strategies — but at the mathematical level the operations are equivalent up to
a handful of config knobs and one optional-`bias` flag. Tilelang additionally
ships a couple of *bundling* kernels (big fused + multilayer recompute) and a
couple of *specialized* kernels (entry-point broadcast, first-layer mix
compute); these are performance/memory optimizations, not additional features.

## Kernel inventory — side by side

| Logical op           | Triton (`mhc_kernel.py`)                                    | Tilelang (`tilelang_kernels/…`)                          | Notes |
| -------------------- | ----------------------------------------------------------- | -------------------------------------------------------- | --- |
| Projection           | `_mhc_projection_fwd_fused` / `_bwd_fused`                  | `norm_fn_kernel._mhc_pre_norm_fn_fwd_mul` / `_fwd_norm`  | Triton stops at `x@Wᵀ` + `ms`. Tilelang fuses RMS. |
| Scale (sigmoid/bias) | `_mhc_scale_fwd_fused` / `_bwd_fused`                       | `pre_split_mixes_kernel._mhc_pre_split_mixes_fwd` / `_bwd` | Same mapping H → (H_pre, H_post, H_res). |
| Sinkhorn             | `_mhc_sinkhorn_fwd_fused[_recompute]` / `_bwd_fused[_recompute]` | `sinkhorn_kernel._mhc_sinkhorn_fwd` / `_bwd`         | Different algorithm, same fixed point. |
| Aggregate            | `_mhc_aggregate_fwd` / `_bwd`                               | `pre_apply_mix_kernel._mhc_pre_apply_mix_fwd` / `_bwd`   | Same math. |
| Expand-combine / Post| `_mhc_expand_combine_fwd` / `_bwd` (+ `_with_bias` variants)| `post_kernel.mhc_post_fwd` / `mhc_post_bwd`              | Same math; triton has `bias`, tilelang doesn't. |
| *Entry broadcast*    | —                                                           | `expand_kernel.expand_to_mhc_fwd` / `_bwd`               | Tilelang-only op: `(s,b,h) → (s,b,n,h)`. |
| *First-layer mix*    | —                                                           | `head_compute_mix_kernel._mhc_head_compute_mix_fwd` / `_bwd` | Specialization of scale — only produces `pre`. |
| *Super-fused pre*    | —                                                           | `pre_big_fuse_kernel._mhc_pre_big_fuse`                  | One kernel = projection + split + sinkhorn + aggregate. |
| *Grad checkpointing* | —                                                           | `multilayer_recompute_kernel.mhc_multilayer_recompute`   | Replays several mHC layers fused for bwd recompute. |

The last four rows are **tilelang-only** — they fuse or specialize the core
ops rather than adding new ones.

---

## Per-op breakdown

### 1. Sinkhorn

|                     | Triton (`_mhc_sinkhorn_*`)                                  | Tilelang (`_mhc_sinkhorn_*`)                               |
| ------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| Input               | `H_res: (s, b, n, n)`, any dtype (cast fp32 internally)     | `H_res: (num_tokens, n, n)`, fp32                          |
| Output              | same shape as input, same dtype                             | same shape, fp32                                           |
| Algorithm           | **Log-space** Sinkhorn. `f = -logsumexp(H + g, cols)`, `g = -logsumexp(H + f, rows)`, `out = exp(f + H + g)` | **Softmax + iterate**. `x = softmax(x, -1) + ε`; then alternating row/col normalize with `+ε` each divide |
| Symmetry            | Symmetric alternation                                       | Asymmetric start (softmax is only along last dim)          |
| Default iterations  | 20                                                          | 10                                                         |
| Exact fixed point?  | Yes, up to fp32 rounding of `exp`                           | No — `+ε` in every step shifts the fixed point slightly    |
| Recompute option    | `_recompute` variant avoids storing full `f/g` history in fwd for memory | Stores `xs` and `sums` in shared mem for bwd recompute |

**Same function?** Yes — both project to the doubly-stochastic manifold. At
finite iteration count they produce different intermediate values, and
tilelang's `+ε` leaves the result ε-away from true doubly-stochastic.

---

### 2. Projection

|                     | Triton (`_mhc_projection_fwd_fused`)                        | Tilelang (`_mhc_pre_norm_fn_fwd_mul` + `_fwd_norm`)        |
| ------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| Input               | `x: (M, nC)`, `phi: (N, nC)` where `M=sb`, `N=2n+n²`        | `x: (num_tokens, n_rms_group·rms_group_size)`, `fn: (mhc_mult3, …)` |
| Output              | `H: (M, 32)` padded (first N valid), `ms: (M,) = mean(x²)`  | Fully RMS-normalized `out: (num_tokens, mhc_mult3)` |
| Computes            | `H = x @ φᵀ`, `ms = mean(x²)` — no RMS applied             | `H = x @ fnᵀ`, then `H · rsqrt(ms + ε)` applied inline; optional `fn := fn · mhc_norm_weight` pre-fold |
| Multi-group?        | No                                                          | Yes — `n_rms_group > 1` computes RMS per group and sums over groups |
| Output padding      | `(M, 32)` (ulp-free bf16 store)                             | `(M, mhc_mult3)` — but internal fragment still 32 cols (reads 32-row `fn`, so the caller must pad) |

**Same function?** Not on their own — tilelang's one kernel equals triton's
**projection → scale** pipeline composed. See next section.

---

### 3. Scale (aka split mixes)

|                     | Triton (`_mhc_scale_fwd_fused`)                             | Tilelang (`_mhc_pre_split_mixes_fwd`)                     |
| ------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| Input               | `H: (M, 32)`, `α: (3,)`, `β: (1, 2n+n²)`, `ms: (M,)`, `n`   | `input_mixes: (num_tokens, 2n+n²)` (already RMS-normed), `mhc_scale: (3,)`, `mhc_base: (2n+n²,)` |
| Step 1 (RMS)        | `rms = sqrt(ms + ε)`, then `H / rms`                        | Not inside this kernel — assumed already applied upstream  |
| Step 2 (affine)     | `H · α + β` (α broadcast to groups of `n, n, n²`)           | `input_mixes · α + β` (same broadcast)                     |
| Step 3 (split)      | `H_pre = sigmoid(· )`, `H_post = 2·sigmoid(·)`, `H_res = · ` (no activation) | `pre = sigmoid(·) + pre_eps`, `post = sigmoid(·) · post_mult_value`, `comb_res = ·` |
| Knobs               | `ε = finfo(fp32).eps`                                        | `pre_eps`, `post_mult_value` are config params             |

**Equivalence**: under `pre_eps = 0`, `post_mult_value = 2`, and a matching
`ε` in the rsqrt, tilelang's (pre_norm_fn + pre_split_mixes) computes the same
function as triton's (projection + scale). All five of the following have to
line up for bit-for-bit parity:

  - tilelang `pre_eps == 0`
  - tilelang `post_mult_value == 2`
  - tilelang `n_rms_group == 1` and `rms_group_size == nC`
  - tilelang `mhc_norm_weight is None`
  - `bias is None` in expand_combine

With those, both pipelines reduce to:

```
rms  = sqrt(mean(x²) + ε)
H    = x @ φᵀ
H_pre  = sigmoid(H[:n]     · α₀ / rms + β[:n])
H_post = 2 · sigmoid(H[n:2n] · α₁ / rms + β[n:2n])
H_res  = H[2n:]           · α₂ / rms + β[2n:]
```

---

### 4. Aggregate / pre-apply-mix

|                     | Triton (`_mhc_aggregate_fwd`)                                | Tilelang (`_mhc_pre_apply_mix_fwd`)                        |
| ------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Input               | `x: (s, b, C, n)`, `H_pre: (s, b, n)`                        | `x: (n_tokens, n, C)` bf16, `mix: (n_tokens, n)` fp32      |
| Output              | `out: (s, b, C)`, same dtype as x                            | `out: (n_tokens, C)` bf16 (hard-cast on store)             |
| Math                | `out[c] = Σₙ x[c,n] · h[n]`                                   | `out[c] = Σₙ x[n,c] · mix[n]` — same sum, different axis layout |

**Same function?** Yes. Differences are purely layout and the bf16 cast on
output.

---

### 5. Expand-combine / Post

|                     | Triton (`_mhc_expand_combine_fwd[_with_bias]`)               | Tilelang (`mhc_post_fwd`)                                  |
| ------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Input               | `f: (s,b,C)`, `bias: (C,)|None`, `H_post: (s,b,n)`, `x: (s,b,C,n)`, `H_res: (s,b,n,n)` | `x: (s,b,C)`, `residual: (s,b,n,C)`, `post_layer_mix: (s,b,n,1)`, `comb_res_mix: (s,b,n,n)` |
| Output              | `out: (s, b, C, n)`                                          | `out: (s, b, n, C)`                                        |
| Math                | `out = (f + bias) ⊗ H_post + x @ H_res`                     | `out = H_post ⊗ f + einsum('mn,mc->nc', H_res, residual)` |
| H_res contraction   | Contracts **first** n dim (`x @ H_res`: `sum_m x[C,m]·H_res[m,n]`) | Contracts **first** n dim (`einsum('abmn,abmc->abnc'…)`) |
| Bias support        | Yes (`_with_bias` kernel)                                    | No — wrapper has no `bias` arg                             |

**Same function?** Yes, when `bias is None`. Triton's `bias` is the one real
feature gap in expand-combine.

---

## Convention & layout cheat sheet

| What                       | Triton                    | Tilelang                  |
| -------------------------- | ------------------------- | ------------------------- |
| `x` in aggregate           | `(s, b, C, n)`            | `(s, b, n, C)` (flattened to `(sb, n, C)`) |
| `x` in expand-combine      | `(s, b, C, n)`            | `(s, b, n, C)`            |
| Output of expand-combine   | `(s, b, C, n)`            | `(s, b, n, C)`            |
| `H_res` first-dim semantic | input stream              | input stream              |
| Sinkhorn shape             | `(s, b, n, n)`            | `(num_tokens, n, n)`      |

Both impls agree on the `H_res` contraction direction, so no transpose is
needed between them — only the `(n, C) ↔ (C, n)` swap when moving x/residual
between sides.

---

## Tilelang-only fusion kernels

### `pre_big_fuse_kernel._mhc_pre_big_fuse`
One kernel that does **the entire "pre" path** in a transformer block:

  1. `_mhc_pre_norm_fn_fwd_norm` — RMS-normalize projected mixes
  2. `_mhc_pre_split_mixes_fwd` — sigmoid / bias / α into pre, post, comb_res
  3. `_mhc_sinkhorn_fwd` — project comb_res to doubly-stochastic
  4. `_mhc_pre_apply_mix_fwd` — aggregate via `pre`

Triton equivalent: four separate kernel launches (projection output + scale +
sinkhorn + aggregate). Same math, 4× fewer launches, all state lives in
register / shared memory across the fused body.

### `multilayer_recompute_kernel.mhc_multilayer_recompute`
Replays multiple mHC layers in one kernel for the backward pass (gradient
checkpointing). Iterates per-token, maintains the residual state in register,
consumes a list of layer pointers. Saves device memory by not storing per-layer
activations; pays per-layer forward cost during bwd.

Triton equivalent: no standalone kernel — caller would rerun the full forward
sequence N times.

### `expand_kernel.expand_to_mhc_fwd`
Broadcasts `(s, b, h) → (s, b, n, h)` (replicates the hidden state across n
streams) at the *entry* of the first mHC block. Backward sums across n.

Triton equivalent: a view + contiguous would do this for free; it's not a
function, it's a layout thing tilelang packages as a kernel for consistency.

### `head_compute_mix_kernel._mhc_head_compute_mix`
A shrunk version of `pre_split_mixes` that only computes the `pre` portion
(width `n`, not `2n + n²`). Used for the first layer where no post / comb_res
mix is needed yet.

Triton equivalent: call full `scale`, ignore 20 of the 24 output columns.

---

## Feature gaps

### Triton-only
- **`bias` in expand_combine.** `_mhc_expand_combine_with_bias_*` kernels fold
  `f ← f + bias` into the outer product without materialising `f + bias`.
  Tilelang wrapper doesn't accept `bias`.
- **Separate projection output.** `mhc_fused_projection` returns raw `(H, ms)`
  — useful if the downstream consumer of `ms` is something other than a scale
  op (e.g. export the statistic for profiling / loss).
- **TF32 toggle.** `use_tf32=False` forces IEEE fp32 for matmul.

### Tilelang-only
- **`mhc_norm_weight`** pre-multiplied into `fn` — fuses an elementwise norm
  weight into the projection without a separate kernel.
- **Multi-group RMS** (`n_rms_group > 1`) — splits the hidden dim into
  independent RMS groups, handy for grouped variants of mHC.
- **`pre_eps`, `post_mult_value`** — config knobs on the sigmoid outputs.
- **Projection / norm split parameter (`n_splits`)** — allows K-splitting the
  projection matmul into parallel chunks combined by a separate norm kernel.
- **Super-fused `pre_big_fuse`** and **`multilayer_recompute`** as above.
- **`expand_to_mhc`** as a standalone kernel.

---

## Optimization strategy differences

Same mathematical work, different performance tactics.

### 1. Autotuning vs hand-tuned configs

**Triton.** Each kernel has a `*_config_fwd()` / `*_config_bwd()` function
that enumerates `{BLOCK_SIZE_M, BLOCK_SIZE_K, STEP_SIZE_K, num_warps,
num_stages}` combinations, wrapped in `@triton.autotune`. First invocation
explores the space and caches the winner. Escape hatch:
`NVTE_DISABLE_TRITON_AUTOTUNING=1` collapses to a single config (first one in
the list) for deterministic per-invocation latency during profiling.

**Tilelang.** No autotuner. Block sizes are baked in as parameters at kernel
JIT time (e.g. `_mhc_post_fwd(mhc, hidden, n_thr=128, h_blk=1024)`). The
`pass_configs` dict passes PTXAS/codegen pragmas straight to the backend:

  - `TL_DISABLE_WARP_SPECIALIZED: True` — turn off Hopper warp-specialized
    pipelining
  - `TL_PTXAS_REGISTER_USAGE_LEVEL: 10` — pin register-usage level (tighter
    than default; lets the author prioritise occupancy vs register pressure)
  - `TL_DISABLE_VECTORIZE_256: True` — cap load/store vectorization at 128b
  - `TL_DISABLE_WGMMA: True` (norm_fn kernels) — fall back to non-WGMMA MMAs

This is a fundamentally different philosophy: triton searches; tilelang
author-picks. Wins and losses are symmetric — autotune adapts to new GPUs
without code changes but costs a cold-start search; hand-tuning is instantly
fast but may miss a new arch's sweet spot.

### 2. Fusion granularity

**Triton.** One-op-per-kernel. Even the `_with_bias` variant is a sibling
kernel, not a parameter path. The triton "pre" sequence is four launches
(projection → scale → sinkhorn → aggregate).

**Tilelang.** Offers both granular and mega-fused versions. `pre_big_fuse`
does all four "pre" ops in one launch, keeping intermediates in registers.
`multilayer_recompute` goes further, running N full mHC layers inside one
kernel for gradient checkpointing. Trade-off: fewer launches and less HBM
traffic, but larger kernels (register and shared-memory pressure) and less
flexibility to mix kernels from different providers.

### 3. Cross-block reduction — atomics vs persistent blocks

**Triton.** Uses `tl.atomic_add` for gradients that reduce across blocks
(`grad_H_pre`, `grad_H_post`, `grad_H_res`, `grad_bias` in expand-combine;
`grad_α`, `grad_β`, `grad_ms` in scale; `grad_phi` piece of projection). This
is **non-deterministic** (atomic ordering varies); `mhc_ops.py` asserts
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=1`.

**Tilelang.** Uses `T.Persistent([…], num_sms, pid)` — a fixed number of
threadblocks each iterating over many tiles — combined with
`T.alloc_reducer(replication='all')` to accumulate within one block.
Cross-block reductions (e.g. `mhc_scale_grad_partial: (num_sms, 3)`) are
written as per-SM partials and summed **outside** the kernel. Deterministic,
but requires sizing `num_sms` to the device (pragma in Python).

### 4. Memory pipelining

**Tilelang.** Explicit:

```python
for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
    T.copy(x[...], xs, disable_tma=True)
    T.copy(xs, xl)
    ...
```

Uses `T.async_copy` (`multilayer_recompute`), opts in/out of TMA per `T.copy`
via `disable_tma`, and invokes `T.pdl_sync()` for Hopper Program-Dependent
Launch. Register vs. shared-mem tiling is under author control.

**Triton.** No explicit async API. The IR lowering and compiler schedule the
loads; `num_stages` in the autotune space is the only user knob and selects
the pipeline depth the compiler targets.

### 5. Shared-memory layout

**Tilelang.** Explicit swizzling annotations:

```python
T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})
```

Selects one of a few known-good bank-conflict-avoiding layouts. Used in
`norm_fn_fwd_mul` and `pre_split_mixes` (for the fragment-layout hint).

**Triton.** Layout is compiler-chosen; no user knob. Works well for standard
matmul shapes; less control over edge cases.

### 6. Precision knobs

Both sides use fp32 accumulators for bf16 matmul. Difference is in how the
cast is expressed:

- **Triton** exposes `use_tf32: bool` (exposed in `mhc_ops.py`) which toggles
  the `precision="tf32"|"ieee"` argument of `tl.dot`. Matters for fp32 inputs.
- **Tilelang** performs the cast explicitly: loads bf16 into a shared-memory
  tile, then `T.copy(bf16_frag, fp32_frag)` before the reduction. No user
  knob; the kernel author pins the precision.

### 7. Determinism

- **Triton forward**: deterministic.
- **Triton backward**: **non-deterministic** (atomics); gated by the
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO` env flag with an `assert` failure if
  unset.
- **Tilelang (fwd and bwd)**: deterministic — uses the persistent-block +
  reducer pattern instead of atomics.

This is a real behavioural difference, not just performance. Run-to-run bit
reproducibility in training is a triton-path concern that tilelang sidesteps.

### 8. Kernel persistence / launch overhead

Tilelang's `T.Persistent` gives each threadblock a work loop over multiple
tiles, so one launch handles many tiles (sized to `num_sms`). Triton launches
one block per tile (standard grid). For kernels where tile count is modest
and per-launch overhead is non-trivial (small-M, small-K cases) the persistent
model wins; for huge problems the overhead is amortised either way.

### 9. Summary

| Strategy dimension           | Triton                                | Tilelang                              |
| ---------------------------- | ------------------------------------- | ------------------------------------- |
| Autotuning                   | Runtime search over config list       | Hand-tuned, baked in at JIT           |
| Fusion granularity           | One op per kernel                     | Per-op + mega-fused (`pre_big_fuse`, `multilayer_recompute`) |
| Cross-block reduction        | `tl.atomic_add` (non-deterministic)   | `T.Persistent` + `alloc_reducer` + per-SM partials (deterministic) |
| Memory pipelining            | Compiler-scheduled                    | Explicit `T.Pipelined` / `T.async_copy`, TMA opt-in/out |
| Shared-memory layout         | Compiler-chosen                       | Explicit swizzle annotations          |
| Precision knobs              | `use_tf32` flag                       | Explicit fp32 cast in kernel          |
| Determinism                  | Non-det in backward                   | Fully deterministic                   |
| Kernel persistence           | Grid = one block per tile             | Persistent blocks iterate over tiles  |

---

## When this matters in practice

If you're deciding which impl to use, the axes that end up mattering:

- **Determinism.** If your training / regression tests require bitwise
  reproducibility, triton's backward is a problem until you either turn off
  autotune *and* sit on a single-SM config (collapsing atomics to a single
  writer), or replace it. Tilelang is deterministic by default.

- **Bias.** If you need fused `f + bias` in expand-combine, triton is the only
  option of the two. Tilelang would need the caller to materialise `f + bias`
  up front.

- **Multi-group RMS / `mhc_norm_weight`.** If your mHC variant needs either,
  tilelang is your only option.

- **Grad checkpointing / memory pressure.** `multilayer_recompute` is unique to
  tilelang. On triton you'd need torch-level activation checkpointing.

- **Cold start cost.** Triton pays an autotune search on first invocation of
  each shape; tilelang pays a single JIT compile (slower initial compile, no
  search). For short-lived jobs or notebooks, triton feels slower to warm up.

- **New GPU architectures.** Triton's autotune picks fresh configs without
  code changes. Tilelang's hand-picked configs may need re-tuning per arch.

Under the config defaults above (`pre_eps=0, post_mult_value=2, n_rms_group=1,
mhc_norm_weight=None, bias=None`) the functional outputs match within bf16
rounding — so feature parity reduces to which engineering trade-offs you prefer.
