# mHC kernel benchmark

Per-kernel GPU time for each (op, direction, framework) on 13 LLM-training-class shapes, profiled by `mhc_bench.sh` → `mhc_bench.py` → nsys, using the kernel-name → (op, framework) mapping in `kernel_comparison.md`.

## Methodology

- **Numbers are `Avg (ns)` from `cuda_gpu_kern_sum`**, averaged across 5 measured iterations per kernel (after 5 warmup iterations per framework for JIT + autotune + cache warming).
- **Per-op sums** follow `kernel_comparison.md`. Notable boundaries:
  - tilelang `proj_scale` = `pre_norm_fn_{mul,norm}` + `pre_split_mixes`
  - triton `proj_scale` = `projection + scale`
  - triton's projection bwd uses a cuBLAS GEMM (`nvjet_sm100_sss_*` + `cublasLt::splitKreduce`) for `grad_phi` when `norm_weight` is None; these are counted in triton's `proj_scale` bwd.
- **Wrapper kernels are excluded**: `at::native::*elementwise*`, `FillFunctor` (incl. the 256 MB L2 flusher), `bfloat16_copy`, generic `reduce_kernel` not attributable to an op, torch RNG kernels.
- **L2 cache is flushed before every op invocation** by zeroing a 256 MB scratch buffer (~2× B200's L2). This forces every kernel to read its inputs cold from HBM and is included in both warmup and measured iters so they run under matching cache conditions.
- **Production dtypes**: H/mix tensors are fp32 on both sides (matches `mhc_fused_projection`'s fp32 output and the downstream consumers). Activation tensors (`x`, `f`, `residual`) are bf16.
- **Gradient seeds are `randn_like` for sinkhorn**, not `ones_like` — a uniform-ones gradient drives tilelang's sinkhorn bwd recompute into denormals (`dy - mean(dy)` style intermediates collapse to zero, subsequent multiplies/divides hit subnormal slow-paths), which made the old sinkhorn-bwd gap unfairly large.
- Each framework is profiled in its own `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` window — running both frameworks under a single multi-range capture crashes the nsys agent and silently drops the second range.
- `--iters 5` means 5 instances per kernel in the averages.
- **Triton autotunes per shape (8–216 configs per kernel, 486 total across 11 kernels). Tilelang doesn't autotune — block sizes / num_warps / num_stages are hardcoded in the kernel source.** This is a structural advantage triton has on every per-shape benchmark; some portion of the gap reflects tilelang shipping a less-tuned point in the design space.
- Clocks are **not locked**. Run-to-run variance on the order of a few % is expected; kernel-level ratios are stable.

## Results

All times in **µs** per (op, direction) call. The `tl/tr` column is **how many times slower tilelang is** compared to triton on that op (e.g. `2.50×` means tilelang takes 2.5× as long as triton; <1× means tilelang is faster).

### B=4, T=4096, C=4096 — LLaMA-2-7B / Mistral-7B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |    127.5 |    252.1 |  1.98× |
| proj_scale | bwd |    494.0 |   1481.1 |  3.00× |
| sinkhorn | fwd |     13.0 |     46.7 |  3.58× |
| sinkhorn | bwd |     20.2 |     44.9 |  2.22× |
| aggregate | fwd |     97.4 |     98.1 |  1.01× |
| aggregate | bwd |    168.8 |    266.1 |  1.58× |
| post | fwd |    170.4 |    174.0 |  1.02× |
| post | bwd |    308.0 |    443.7 |  1.44× |
| **pipeline** | **fwd+bwd** | **  1399.4** | **  2806.6** | ** 2.01×** |

### B=2, T=4096, C=5120 — LLaMA-2-13B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     87.1 |    158.6 |  1.82× |
| proj_scale | bwd |    318.0 |    852.7 |  2.68× |
| sinkhorn | fwd |      9.7 |     24.8 |  2.56× |
| sinkhorn | bwd |     12.3 |     23.6 |  1.91× |
| aggregate | fwd |     62.5 |     63.8 |  1.02× |
| aggregate | bwd |    108.3 |    178.3 |  1.65× |
| post | fwd |    107.9 |    111.6 |  1.03× |
| post | bwd |    194.4 |    282.6 |  1.45× |
| **pipeline** | **fwd+bwd** | **   900.4** | **  1695.9** | ** 1.88×** |

### B=1, T=4096, C=6656 — LLaMA-2-33B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     59.6 |    104.0 |  1.75× |
| proj_scale | bwd |    209.4 |    428.5 |  2.05× |
| sinkhorn | fwd |      7.8 |     14.0 |  1.80× |
| sinkhorn | bwd |     10.1 |     22.5 |  2.22× |
| aggregate | fwd |     42.0 |     44.1 |  1.05× |
| aggregate | bwd |     69.4 |    129.9 |  1.87× |
| post | fwd |     71.8 |     80.3 |  1.12× |
| post | bwd |    128.0 |    186.1 |  1.45× |
| **pipeline** | **fwd+bwd** | **   598.0** | **  1009.5** | ** 1.69×** |

### B=1, T=4096, C=8192 — LLaMA-2-70B / Qwen-72B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     74.1 |    125.8 |  1.70× |
| proj_scale | bwd |    254.8 |    425.1 |  1.67× |
| sinkhorn | fwd |      7.7 |     14.1 |  1.83× |
| sinkhorn | bwd |     10.2 |     22.4 |  2.21× |
| aggregate | fwd |     51.1 |     52.7 |  1.03× |
| aggregate | bwd |     85.3 |    149.4 |  1.75× |
| post | fwd |     87.1 |     93.4 |  1.07× |
| post | bwd |    156.7 |    224.6 |  1.43× |
| **pipeline** | **fwd+bwd** | **   726.9** | **  1107.4** | ** 1.52×** |

### B=4, T=8192, C=4096 — LLaMA-3-8B (8k ctx, B=4)

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |    231.8 |    438.9 |  1.89× |
| proj_scale | bwd |    950.2 |   2966.8 |  3.12× |
| sinkhorn | fwd |     21.3 |     90.2 |  4.24× |
| sinkhorn | bwd |     32.4 |     86.8 |  2.68× |
| aggregate | fwd |    189.6 |    190.5 |  1.00× |
| aggregate | bwd |    336.9 |    505.9 |  1.50× |
| post | fwd |    335.7 |    341.4 |  1.02× |
| post | bwd |    611.0 |    870.1 |  1.42× |
| **pipeline** | **fwd+bwd** | **  2708.9** | **  5490.6** | ** 2.03×** |

### B=1, T=8192, C=4096 — LLaMA-3-8B (8k ctx, B=1)

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     75.1 |    129.6 |  1.72× |
| proj_scale | bwd |    252.8 |    737.5 |  2.92× |
| sinkhorn | fwd |      9.6 |     24.8 |  2.58× |
| sinkhorn | bwd |     12.4 |     23.5 |  1.90× |
| aggregate | fwd |     50.7 |     51.5 |  1.02× |
| aggregate | bwd |     86.0 |    146.5 |  1.70× |
| post | fwd |     87.2 |     90.1 |  1.03× |
| post | bwd |    156.6 |    230.5 |  1.47× |
| **pipeline** | **fwd+bwd** | **   730.5** | **  1433.9** | ** 1.96×** |

### B=4, T=8192, C=7168 — DeepSeek-V2-Lite

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |    374.0 |    741.8 |  1.98× |
| proj_scale | bwd |   1622.3 |   3270.5 |  2.02× |
| sinkhorn | fwd |     21.3 |     90.2 |  4.23× |
| sinkhorn | bwd |     32.4 |     86.8 |  2.68× |
| aggregate | fwd |    323.6 |    324.6 |  1.00× |
| aggregate | bwd |    588.9 |    874.3 |  1.48× |
| post | fwd |    582.6 |    600.8 |  1.03× |
| post | bwd |   1065.2 |   1490.5 |  1.40× |
| **pipeline** | **fwd+bwd** | **  4610.2** | **  7479.6** | ** 1.62×** |

### B=1, T=8192, C=8192 — LLaMA-3-70B (8k ctx)

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |    124.8 |    245.6 |  1.97× |
| proj_scale | bwd |    484.7 |    830.8 |  1.71× |
| sinkhorn | fwd |      9.6 |     24.8 |  2.58× |
| sinkhorn | bwd |     12.7 |     23.5 |  1.85× |
| aggregate | fwd |     97.8 |     99.1 |  1.01× |
| aggregate | bwd |    168.4 |    270.8 |  1.61× |
| post | fwd |    170.3 |    179.0 |  1.05× |
| post | bwd |    307.9 |    440.3 |  1.43× |
| **pipeline** | **fwd+bwd** | **  1376.3** | **  2113.9** | ** 1.54×** |

### B=1, T=8192, C=16384 — LLaMA-3-405B (8k ctx)

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |    218.1 |    475.8 |  2.18× |
| proj_scale | bwd |    938.9 |   2100.7 |  2.24× |
| sinkhorn | fwd |      9.6 |     24.8 |  2.59× |
| sinkhorn | bwd |     12.8 |     23.5 |  1.84× |
| aggregate | fwd |    190.0 |    197.5 |  1.04× |
| aggregate | bwd |    336.6 |    522.9 |  1.55× |
| post | fwd |    336.0 |    359.0 |  1.07× |
| post | bwd |    610.9 |    847.3 |  1.39× |
| **pipeline** | **fwd+bwd** | **  2652.8** | **  4551.6** | ** 1.72×** |

### B=8, T=2048, C=2560 — GPT-3-1.3B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     86.6 |    165.5 |  1.91× |
| proj_scale | bwd |    319.9 |   1464.9 |  4.58× |
| sinkhorn | fwd |     13.0 |     46.6 |  3.57× |
| sinkhorn | bwd |     20.3 |     44.8 |  2.21× |
| aggregate | fwd |     62.5 |     63.4 |  1.01× |
| aggregate | bwd |    108.6 |    178.3 |  1.64× |
| post | fwd |    108.1 |    113.0 |  1.05× |
| post | bwd |    194.5 |    296.9 |  1.53× |
| **pipeline** | **fwd+bwd** | **   913.6** | **  2373.4** | ** 2.60×** |

### B=4, T=2048, C=4096 — GPT-3-6.7B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     75.4 |    129.7 |  1.72× |
| proj_scale | bwd |    253.1 |    736.8 |  2.91× |
| sinkhorn | fwd |      9.6 |     24.9 |  2.61× |
| sinkhorn | bwd |     12.7 |     23.6 |  1.85× |
| aggregate | fwd |     51.0 |     51.2 |  1.00× |
| aggregate | bwd |     86.0 |    146.3 |  1.70× |
| post | fwd |     87.5 |     89.9 |  1.03× |
| post | bwd |    156.8 |    230.5 |  1.47× |
| **pipeline** | **fwd+bwd** | **   732.0** | **  1432.8** | ** 1.96×** |

### B=1, T=2048, C=12288 — GPT-3-175B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     56.2 |    182.0 |  3.24× |
| proj_scale | bwd |    197.2 |    440.5 |  2.23× |
| sinkhorn | fwd |      7.2 |      8.2 |  1.14× |
| sinkhorn | bwd |      9.9 |     22.2 |  2.24× |
| aggregate | fwd |     39.7 |     45.4 |  1.14× |
| aggregate | bwd |     66.5 |    124.9 |  1.88× |
| post | fwd |     66.6 |     76.9 |  1.15× |
| post | bwd |    118.9 |    182.0 |  1.53× |
| **pipeline** | **fwd+bwd** | **   562.3** | **  1082.1** | ** 1.92×** |

### B=3, T=8192, C=2048 — Nemotron 2B

| op | dir | triton (µs) | tilelang (µs) | tl/tr |
|---|---|---:|---:|---:|
| proj_scale | fwd |     99.3 |    200.5 |  2.02× |
| proj_scale | bwd |    377.1 |   2142.5 |  5.68× |
| sinkhorn | fwd |     18.1 |     68.5 |  3.79× |
| sinkhorn | bwd |     27.8 |     66.1 |  2.37× |
| aggregate | fwd |     74.5 |     74.5 |  1.00× |
| aggregate | bwd |    126.8 |    205.1 |  1.62× |
| post | fwd |    129.0 |    129.5 |  1.00× |
| post | bwd |    232.1 |    372.3 |  1.60× |
| **pipeline** | **fwd+bwd** | **  1084.8** | **  3258.9** | ** 3.00×** |

## Geomean: how many times slower tilelang is

Geometric mean of per-config `tl/tr` ratios. `3.00×` means tilelang takes 3× as long as triton on that op on average; `<1×` would mean tilelang is faster.

| op | dir | tl/tr (geomean) |
|---|---|---:|
| proj_scale | fwd |  1.96× |
| proj_scale | bwd |  2.66× |
| sinkhorn | fwd |  2.68× |
| sinkhorn | bwd |  2.15× |
| aggregate | fwd |  1.03× |
| aggregate | bwd |  1.65× |
| post | fwd |  1.05× |
| post | bwd |  1.46× |

## Whole-pipeline: how many times slower tilelang is

Sum of all (op, fwd+bwd) times per config, then ratio, then geomean across configs.

- tilelang vs triton (full pipeline): **1.92× slower**

