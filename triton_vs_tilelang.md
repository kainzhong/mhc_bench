# Triton vs. Tilelang mHC — wrapped API comparison

Compares the two mHC (manifold Hyper-Connection) implementations at the
`torch.autograd.Function`-backed wrapper layer each one exposes:

- Triton: `triton_kernels.mhc_ops`
- Tilelang: `tilelang_kernels.modeling.mhc.ops.ops`

## 1. Op inventory — what both sides ship vs. what only one side ships

| Logical op in an mHC block | Triton wrapper | Tilelang wrapper |
| --- | --- | --- |
| Projection (`x @ phi.T`, `ms = mean(x²)`) | `mhc_fused_projection` | `mhc_pre_norm_fn`\* |
| RMS normalize (`· rsqrt(ms+eps)`) | fused into `mhc_fused_scale` | fused into `mhc_pre_norm_fn`\* |
| Split + sigmoid + bias → `h_pre, h_post, h_res` | `mhc_fused_scale` | `mhc_pre_split_mixes` |
| Sinkhorn (→ doubly-stochastic) | `mhc_fused_sinkhorn` | `sinkhorn_normalize` |
| Aggregate `n` streams → 1 | `mhc_fused_aggregate` | `mhc_pre_apply_mix` |
| Expand + combine (`f⊗H_post + x@H_res`) | `mhc_fused_expand_combine` | `mhc_post` |
| First-layer partial scale (pre-mix only) | — | `mhc_head_compute_mix` |
| `(s,b,h) → (s,b,n,h)` broadcast | — | `expand_to_mhc` |
| Mega-fused "pre" path (4 ops in 1 kernel) | — | `mhc_pre_big_fuse` *(inference-only — no autograd backward)* |
| Multi-layer replay for grad checkpointing | — | `mhc_multilayer_recompute` |

\*`mhc_pre_norm_fn` = "projection + RMS normalize" in one call. The closest
triton sequence is `mhc_fused_projection` *then* the RMS part of
`mhc_fused_scale` — there is no standalone triton call that maps 1:1.

**Shared ops (both sides):** projection, RMS, split/scale, sinkhorn,
aggregate, expand-combine.

**Tilelang-only:** `mhc_head_compute_mix`, `expand_to_mhc`,
`mhc_pre_big_fuse`, `mhc_multilayer_recompute`.

**Triton-only:** `bias` parameter inside `mhc_fused_expand_combine`
(`f ← f + bias` folded into the outer product without materializing
`f + bias`). Tilelang's `mhc_post` has no bias arg.

---

## 2. Parameter-by-parameter mapping

With default knobs aligned, both pipelines compute the same math up to
bf16 rounding. The main difference is where each side draws the boundary
between consecutive kernels (see §3).

### 2.1 Projection

| Triton `mhc_fused_projection(x, phi, use_tf32)` | Tilelang `mhc_pre_norm_fn(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps, fuse_grad_acc, n_splits)` |
| --- | --- |
| `x: (M, K)` any dtype | `residual: (..., mhc_mult, hidden_size)` bf16; flattens to `(M, K)` where `K = mhc_mult·hidden_size` |
| `phi: (24, K)` any dtype | `mhc_fn: (24, K)` fp32 |
| — | `mhc_norm_weight: (K,)` fp32 or `None` — pre-multiplied into `mhc_fn` via `_MHCFnNormwMerge` before the projection |
| — | `mhc_norm_eps: float` — RMS ε (triton keeps this in `mhc_fused_scale`) |
| — | `fuse_grad_acc: bool` — bwd-path optimization (no effect on output math); see §5 for the mechanism. Short version: lets this bwd's `x_grad` and `mhc_post` bwd's `d_residual` land in one shared buffer summed on-chip, skipping a torch-level add |
| — | `n_splits: int` — split-K hint (shipped wrapper pins to 1) |
| `use_tf32: bool` | implicit — kernel rounds `mhc_fn` via `round_to_tf32()` and keeps bf16 inputs |
| Returns `(H: (M, 32), ms: (M,))` | Returns `out: (..., 24)` — already RMS-normalized: `(x @ fn.T) · rsqrt(mean(x²) + eps)` |

**Same output?** Yes on the shared math, modulo where the boundary is cut.

- Raw `H = x @ phi.T` is the same matmul on both sides; up to tf32 rounding
  in the MMA the two produce the same `H`. The triton kernel exposes it
  directly; the tilelang kernel computes it internally but doesn't return
  it.
- `ms` (triton) and `rms = sqrt(ms + eps)` (tilelang, applied inline) are
  the same statistic in different forms — `ms = mean(x²)`; tilelang
  divides by `rsqrt(ms + eps)` inside the same kernel.
- Because there's no learnable RMSNorm weight in this layer (unless the
  caller passes `mhc_norm_weight`), composing triton's `mhc_fused_projection`
  + the RMS step of `mhc_fused_scale` produces the same normalized tensor
  as `mhc_pre_norm_fn`, to bf16 rounding.

So the practical answer: the outputs match — just pulled out at different
points in the pipeline.

### 2.2 Scale / split mixes

| Triton `mhc_fused_scale(H, alpha, beta, ms, n)` | Tilelang `mhc_pre_split_mixes(input_mixes, mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps)` |
| --- | --- |
| `H: (M, 32)` — raw projection output | `input_mixes: (s, b, 2n+n²)` — already RMS-normed |
| `alpha: (3,)` | `mhc_scale: (3,)` |
| `beta: (1, 2n+n²)` | `mhc_base: (2n+n²,)` |
| `ms: (M,)` — RMS applied inside this kernel | — (applied upstream in `mhc_pre_norm_fn`) |
| `n: int` | `mhc_mult: int` |
| hardcoded `h_post = 2 · sigmoid(...)` | `mhc_post_mult_value: float` — configurable, pass `2.0` to match triton |
| hardcoded `h_pre = sigmoid(...)` | `mhc_pre_eps: float` — extra ε *added* to the `h_pre` sigmoid output; pass `0.0` to match triton |
| Returns `(h_pre, h_post, h_res)` with shapes `(M, n), (M, n), (M, n²)` | Returns `(pre, post, comb_res)` with shapes `(s, b, n, 1), (s, b, n, 1), (s, b, n, n)` |

**Same output?** Yes with `mhc_pre_eps=0` and `mhc_post_mult_value=2.0` —
the math is identical and only the output layout differs (2D vs 4D with
trailing 1). Any other `pre_eps`/`post_mult_value` puts them on different
functions.

### 2.3 Sinkhorn

| Triton `mhc_fused_sinkhorn(H_res, n, recompute_hist, iters)` | Tilelang `sinkhorn_normalize(x, repeat, eps)` |
| --- | --- |
| `H_res: (s, b, n, n)` any dtype (cast to fp32 internally) | `x: (..., n, n)` **fp32 only** (kernel asserts) |
| `n: int = 4` | inferred from `x.shape[-1]` |
| `iters: int = 20` | `repeat: int = 10` |
| `recompute_hist: bool = True` — opt out of storing fwd history | always recomputes |
| — | `eps: float = 1e-6` — `+eps` in every row/col divide (shifts fixed point by O(ε)) |
| Algorithm: **log-space** alternating logsumexp | Algorithm: **softmax init** then iterative `m / (rowsum+eps)` and `m / (colsum+eps)` |

**Same output?** Yes at the limit, but not bit-identical at finite iters.

Sinkhorn-Knopp converges to the *unique* doubly-stochastic matrix closest
to the input (in KL divergence) regardless of whether you implement it in
log-space or matrix form — so mathematically both sides have the same
fixed point. Two sources of runtime divergence:

- **Finite iteration count.** Defaults differ (triton 20, tilelang 10) and
  at 20 iters neither is bit-converged. Intermediate values differ.
- **tilelang's `+ε`** in every `1/(sum+ε)` divide perturbs the fixed point
  by O(ε); triton's log-space form has no such offset.

With matched `iters`/`repeat` and small `ε`, outputs agree to ~2e-3
absolute at bf16 — fine for the bf16 tolerance bar, not close enough for
bit-level reproducibility.

### 2.4 Aggregate

| Triton `mhc_fused_aggregate(x, H_pre, n, use_tf32)` | Tilelang `mhc_pre_apply_mix(x, mix, out)` |
| --- | --- |
| `x: (s, b, C, n)` | `x: (..., n, C)` — **axis order swapped** |
| `H_pre: (s, b, n)` any dtype | `mix: (..., n, 1)` fp32 with trailing singleton |
| `n: int` | inferred |
| `use_tf32: bool` | implicit |
| — | `out: Tensor \| None` — write-in buffer; wrapper allocates bf16 if `None` |
| Returns `out: (s, b, C)` same dtype as `x` | Returns `out: (..., C)` bf16 (hard-cast) |

**Same output?** Yes, modulo the axis-order reshuffle and the bf16 hard-cast
on the tilelang side. Math is identical; the correctness harness sees 0.0
abs diff when `x` is already bf16.

### 2.5 Expand-combine / mhc_post

| Triton `mhc_fused_expand_combine(f, bias, H_post, x, H_res, n, use_tf32)` | Tilelang `mhc_post(x, residual, post_layer_mix, comb_res_mix, out)` |
| --- | --- |
| `f: (s, b, C)` — sub-layer output | `x: (s, b, C)` — same role, different name |
| `bias: (C,) \| None` — fused `f + bias` | **not supported** |
| `H_post: (s, b, n)` any dtype | `post_layer_mix: (s, b, n, 1)` fp32 |
| `x: (s, b, C, n)` — hyper-conn residual | `residual: (s, b, n, C)` — **axis order swapped** |
| `H_res: (s, b, n, n)` | `comb_res_mix: (s, b, n, n)` fp32 |
| `n: int`, `use_tf32: bool` | both implicit |
| — | `out: Tensor \| None` |
| Returns `(s, b, C, n)` | Returns `(s, b, n, C)` |

**Same output?** Yes with `bias=None`, modulo the axis-order reshuffle.
Math is identical and the correctness harness sees 0.0 abs diff at bf16.
With `bias≠None` only triton can produce it — tilelang's `mhc_post` would
require the caller to materialise `x + bias` first, which drops the whole
point of the fused path.

---

## 3. Fusion boundary — what actually differs between "same op"

Both impls compute the same "pre" path, but the cut between consecutive
kernels is different:

```
triton:   x → [proj: (H, ms)] → [scale + RMS: h_pre, h_post, h_res] → [sinkhorn] → [aggregate]
tilelang: x → [proj + RMS: H] → [split: pre, post, comb_res]          → [sinkhorn] → [pre_apply_mix]
                        ^^^  RMS moved one kernel upstream
```

Consequences:

- Triton surfaces `ms` — you can log it, reuse it for something else, or
  run it through your own scale.
- Tilelang consumes `ms` internally and never returns it. There is no
  way to pull the pre-RMS matmul output out of `mhc_pre_norm_fn`.
- Unit-level benchmarking between the two should compare at each impl's
  natural boundary rather than forcing identical op coverage. Attempting
  `proj_only` on the tilelang side is not well-defined.

Under the aligned defaults (`mhc_pre_eps = 0`, `mhc_post_mult_value = 2`,
`mhc_norm_weight = None`, `bias = None`, single RMS group, same `eps`) both
end-to-end pipelines compute the same function modulo:

- Sinkhorn algorithm difference (log-space vs softmax+ε).
- bf16 rounding across the matmul reductions.
- `round_to_tf32` applied to `mhc_fn` inside the tilelang norm_fn kernel.

---

## 4. PyTorch-surface differences beyond the math

| Concern | Triton | Tilelang |
| --- | --- | --- |
| Autograd surface | `autograd.Function` subclass per op | `autograd.Function` subclass per op |
| `bias` in expand-combine | Yes | No |
| `ms` / RMS stat observable | Yes (`mhc_fused_projection` returns it) | No |
| Multi-group RMS (`n_rms_group > 1`) | No | Latent — kernel IR supports it, wrapper pins to 1 and doesn't expose it. See sub-section below |
| Pre-multiplied `mhc_norm_weight` | No | `_MHCFnNormwMerge` folds it into `mhc_fn` |
| `main_grad` write-through | No | `_MHCFnNormwMerge` writes into `fn.main_grad` / `normw.main_grad` when present (Megatron-style fp32 grad-accum buffer) — see §5 |
| Entry broadcast `(s,b,h) → (s,b,n,h)` | Do it with `view`/`expand`/`contiguous` | `expand_to_mhc` dedicated op |
| First-layer partial scale | Call full `mhc_fused_scale`, ignore 20 of 24 columns | `mhc_head_compute_mix` — only computes pre-mix |
| Mega-fused pre path | No | `mhc_pre_big_fuse` — projection + split + sinkhorn + aggregate in one launch; **inference-only (no backward)** — called only under `if not torch.is_grad_enabled()` |
| Multi-layer grad checkpointing | At the torch level | `mhc_multilayer_recompute` as a dedicated op |
| Precision knob | `use_tf32: bool` per-op flag | Implicit; `round_to_tf32` applied in the wrapper, bf16/fp32 split is baked in |
| Backward determinism | **Non-deterministic** (atomics); gated by `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` | **Deterministic** — persistent-block + reducer + per-SM partial accumulation |
| Sinkhorn memory/compute trade | `recompute_hist: bool` flag | Always recomputes |
| Cold start | Autotune search on first shape (`NVTE_DISABLE_TRITON_AUTOTUNING=1` to pin) | TileLang JIT compile on first shape config |
| `sum().backward()` (stride-0 grad) | Accepted | Rejected by some kernels (sinkhorn bwd, norm_fn bwd); use `.backward(torch.ones_like(out))` |
| `mhc_fn` padding requirement | N/A — `phi: (24, K)` is all the kernel reads | `mhc_fn` underlying storage must have **32 rows** (the projection kernel reads rows 0–31 even though `mhc_mult3 = 24`). Allocate `(32, K)` and pass `.view()[:24]`; the wrapper asserts `(24, K)` but doesn't pad for you. |
| `n_splits` parameter on `mhc_pre_norm_fn` | N/A | Advertised in the signature but **inert** — wrapper body overrides to 1 (comment: "TileLang doesn't support split-K") |
| Input dtype strictness | Wrappers auto-cast internally | `mhc_post` asserts `x, residual: bf16` and `mixes: fp32`; `mhc_pre_norm_fn` asserts `x: bf16, fn: fp32` — callers must pre-cast |
| `mhc_post` bwd hand-off housekeeping | N/A | `mhc_post_bwd` always stashes `d_residual` on `residual.storage().grad_from_mhc_post` (default `fuse_grad_acc=True`). Downstream `mhc_pre_norm_fn`/`mhc_pre_apply_mix` bwd `del`s it. If nothing downstream consumes, it leaks on the storage for that tensor's lifetime. |


> ### Multi-group RMS — what `n_rms_group` would enable

Not user-visible today — the wrapper calls the kernel with
`n_rms_group=1`. Flagging it because the parameter exists in the kernel
signature and may show up if the wrapper is extended later.

Triton's RMSNorm step uses `ms: (M,)` — one mean-square per token across the
entire hidden dim `K = mhc_mult · hidden_size`:

```
ms = mean(x[m, :]²)
rms = sqrt(ms + eps)
H_normed = (x @ phi.T) / rms
```

The tilelang `mhc_pre_norm_fn` kernel parameterizes the hidden axis as
`n_rms_group · rms_group_size`. With `n_rms_group > 1`, the hidden axis
is sliced into groups and each group gets its own RMS denominator:

```
for k in range(n_rms_group):
    x_k   = x[:, k*G : (k+1)*G]
    fn_k  = fn[:, k*G : (k+1)*G]
    ms_k  = mean(x_k²)
    rms_k = sqrt(ms_k + eps)
    out  += (x_k @ fn_k.T) / rms_k
```

Equivalently: split `x` along the hidden axis, RMSNorm each slice with its
own statistic, then the per-group matmuls sum into the same output (the
matmul is linear over the hidden axis, so per-group scales just factor in).

Intended use would be **per-stream normalization** — setting
`n_rms_group = mhc_mult` and `rms_group_size = hidden_size` gives each of
the `n` mHC streams its own RMS, so one loud stream doesn't swamp the
normalization of the others. `n_rms_group = 1` collapses to standard
single-group RMSNorm and matches triton's path.

---

## 5. Optimization strategy differences

| Dimension | Triton | Tilelang |
| --- | --- | --- |
| Config selection | `@triton.autotune` over block / warps / stages list, cached per shape | Hand-tuned configs baked into `@tilelang.jit` wrappers; `pass_configs` overrides PTXAS/codegen (warp specialize, register-usage level, WGMMA, 256-bit vec) |
| Fusion granularity | One op = one kernel (`_with_bias` is a sibling kernel, not a parameter path) | Granular ops + mega-fused kernels (`pre_big_fuse`, `multilayer_recompute`) |
| Cross-block reduction | `tl.atomic_add` — non-deterministic | `T.Persistent([...], num_sms, pid)` + `T.alloc_reducer(replication='all')` + per-SM partials summed outside kernel — deterministic |
| Memory pipelining | Compiler-scheduled; only user knob is `num_stages` in autotune space | Explicit `T.Pipelined(...)`, `T.async_copy`, TMA opt-in/out per `T.copy` |
| Shared-memory layout | Compiler-chosen | Explicit `tilelang.layout.make_swizzled_layout` annotations |
| Precision path | `use_tf32` flag toggles `tl.dot(precision='tf32'|'ieee')` | Explicit casts `T.copy(bf16_frag, fp32_frag)`; `round_to_tf32` in python wrapper |
| Kernel persistence | Grid = one block per tile | `T.Persistent` — persistent blocks loop over tiles, sized to device SM count |
| Gradient accumulation | bwd returns `grad` tensors, torch adds to `param.grad` | `fuse_grad_acc=True` chains one op's `d_residual` into the next op's bwd kernel as a write-in buffer (see sub-section below); `_MHCFnNormwMerge` writes into `fn.main_grad` directly when that attribute is present |
| Recompute-vs-store | Sinkhorn has `recompute_hist` flag; projection/aggregate always store; expand-combine always stores | Sinkhorn always recomputes; whole-layer replay via `mhc_multilayer_recompute` |

> ### The `fuse_grad_acc` chain, in detail

Output math is the same either way — this is purely a bwd-path optimization
to avoid a torch-level add.

An mHC block consumes the *same* `residual` tensor twice during forward:
once at the entry (`mhc_pre_norm_fn(residual, fn)`) and once at the exit
(`mhc_post(x, residual, ...)`). In backward, both ops need to produce a
gradient w.r.t. that shared tensor and those grads have to be summed.

Stock autograd would allocate two separate grad tensors and add them via a
torch-level op. Tilelang's `fuse_grad_acc` path skips that add:

1. `mhc_post_bwd(...)` computes `d_residual` and stashes it on the tensor's
   storage:
   ```python
   residual.untyped_storage().grad_from_mhc_post = d_residual
   ```
2. When `mhc_pre_norm_fn` bwd fires later, it picks that stashed tensor up
   and uses it directly as its own `x_grad` output buffer:
   ```python
   x_grad = x.untyped_storage().grad_from_mhc_post.view_as(x)
   ```
3. The `_mhc_pre_norm_fn_bwd_mul` kernel loads the existing contents of
   `x_grad` into a fragment, accumulates its own contribution on top
   (`T.gemm(..., clear_accum=False)` plus the RMS-derivative term), and
   writes the sum back — all on-chip, in the same store that would have
   happened anyway.
4. The autograd return then passes `None` for the residual grad slot so
   torch does not attempt to add a second time.

Net effect: one buffer, summed in the last store of the kernel, no
zero-init, no torch-level add, no second allocation. The same mechanism
also appears in `mhc_pre_apply_mix`'s backward (it checks
`hasattr(x.untyped_storage(), 'grad_from_mhc_post')` and folds its own
`x_grad` into that same stash when the upstream `mhc_post` populated it).

Triton has no equivalent — gradients flow through `.grad` like any other
autograd function, and the add happens at the torch level.

> ### `main_grad` write-through — same idea, for weight params

Megatron-LM attaches a second gradient buffer to each learnable parameter
called `main_grad` — a pre-allocated fp32 tensor used to accumulate
parameter gradients across microbatches in higher precision than the
parameter dtype (bf16/fp16). The normal flow is:

```
bwd kernel → fresh grad tensor → torch adds to .grad → loop casts & adds to .main_grad
```

Three passes over the grad data, plus an extra allocation.

Tilelang's `_MHCFnNormwMerge` (the autograd fn that folds
`mhc_norm_weight` into `mhc_fn` before the projection) sniffs for
`main_grad` on its inputs:

```python
ctx.fn_main_grad    = getattr(fn, 'main_grad', None)
ctx.normw_main_grad = getattr(normw, 'main_grad', None)
```

If present, the bwd kernel receives `main_grad` directly as its output
buffer and accumulates into it in-place (same `clear_accum=False`
pattern). The autograd fn returns `None` for that slot so torch doesn't
allocate a second grad and double-count.

```
bwd kernel → .main_grad (fp32, directly)
```

One pass instead of three. Same underlying optimization as
`fuse_grad_acc`, just targeting weight-parameter gradients via Megatron's
convention instead of chaining activation gradients between adjacent
mHC ops.

Triton has no equivalent — parameter grads flow through `.grad` and the
training loop is responsible for syncing them into `main_grad` later.

> ### Multi-layer replay — `mhc_multilayer_recompute`

Activation-checkpointing primitive: replays the mHC plumbing across **N
transformer layers in a single kernel**, keeping the residual in
registers across layer boundaries instead of round-tripping it through
HBM.

The mHC glue per layer is `layer_input = aggregate(residual, pre_mix)`
before the sublayer, and `new_residual = post_mix ⊗ layer_output +
comb_mix @ residual` after it. Those are what the kernel replays. It
does **not** replay the sublayer itself (attention / FFN) — `layer_output`
is loaded from the saved forward. It also doesn't replay the projection /
split / sinkhorn — those mixes are assumed saved.

Per-token CTA, for each of the `N` layers:

1. `residual: (mhc, h_blk)` stays in a register fragment across all
   layers; only the first layer loads it from HBM.
2. Next layer's `pre_mix`, `post_mix`, `comb_mix`, `layer_output` are
   `T.async_copy`-ed into a double-buffered shared slot (`[2, ...]`,
   `phase = i_layer % 2`), overlapping HBM load of layer `i+1` with
   compute of layer `i`.
3. `layer_input` and `new_residual` are stored to HBM each layer
   (the sublayer bwd and the post bwd need them).
4. `residual_register ← bf16(new_residual)` — bf16 round-trip keeps the
   recompute bit-identical to the non-checkpointed forward (which would
   have stored bf16 to HBM between layers).

The wrapper takes lists of per-layer pointers and builds a device-side
pointer table once (`_make_ptr_tables_batched`, pinned-CPU staging →
one HtoD copy), so the kernel just indexes into arrays of pointers.

Versus `torch.utils.checkpoint`: that re-runs the whole python-level
forward — dozens of kernel launches × N layers, residual written to
HBM and read back each layer, no async pipelining. This kernel collapses
it to one launch with the residual staying in registers.

Triton has no equivalent — you'd pay the full `torch.utils.checkpoint`
cost or save every residual.

### Philosophy

- **Triton** searches the config space at runtime: adapts to new GPUs without
  code changes, pays a cold-start autotune cost on first call per shape.
- **Tilelang** pins configs by hand: instantly fast on the arch it was tuned
  for, may need re-tuning per arch, but has headroom to express Hopper-
  specific primitives (TMA, WGMMA, async_copy, persistent blocks).
- **Triton** writes fewer, simpler kernels and lets the compiler schedule.
  **Tilelang** writes more, hand-scheduled kernels and reaches for
  explicit swizzling, warp specialization, and mega-fusion.
- **Triton** prioritises author throughput. **Tilelang** prioritises
  end-to-end fusion and determinism.

---

## 6. When it matters in practice

- Need `bias` fused into expand-combine → **triton** is the only option.
- Need `mhc_norm_weight` folded into the projection matmul → **tilelang**.
- Need bitwise-reproducible backward → **tilelang**.
- Need grad checkpointing inside one or several mHC layers →
  **tilelang** (`mhc_multilayer_recompute`).
- Need to observe or re-use `ms` → **triton** (`mhc_fused_projection`).
- Care about minimising launch overhead for the pre path → **tilelang**
  (`mhc_pre_big_fuse` collapses 4 ops into 1 launch).
- Running on a brand-new architecture where no one has hand-tuned yet →
  **triton** (autotune will pick fresh configs without code changes).
