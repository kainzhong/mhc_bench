# mHC kernel benchmark

Per-kernel GPU time for each (op, direction, framework) on 13 LLM-training-class shapes, profiled by `mhc_bench.sh` → `mhc_bench.py` → nsys, using the kernel-name → (op, framework) mapping in `kernel_comparison.md`.

## Methodology

- **Numbers are `Avg (ns)` from `cuda_gpu_kern_sum`**, averaged across 5 measured iterations per kernel (after 5 warmup iterations per framework for JIT + autotune + cache warming).
- **Per-op sums** follow `kernel_comparison.md`. Notable boundaries:
  - tilelang `proj_scale` = `pre_norm_fn_{mul,norm}` + `pre_split_mixes`
  - triton `proj_scale` = `projection + scale`
  - cutile `proj_scale` = `proj_rms` only (no scale kernel)
- **Wrapper kernels are excluded**: `at::native::*elementwise*`, `FillFunctor`, `bfloat16_copy`, `reduce_kernel`, `cublasLt::splitKreduce`, `nvjet_sm100_tst_*`, torch RNG kernels.
- Each framework is warmed up and profiled in its own `cudaProfilerStart/Stop` window (via `--capture-range-end=repeat`) so L2/cache state doesn't leak between frameworks.
- `--iters 5` means 5 instances per kernel in the averages.
- Clocks are **not locked**. Run-to-run variance on the order of a few % is expected; kernel-level ratios are stable.

## Results

All times in **µs** per (op, direction) call. `tr vs cu` / `tr vs tl` columns are **how many times faster triton is** compared to cutile / tilelang on that op (e.g. `2.50×` means triton is 2.5× faster; <1× means triton is slower).

### B=4, T=4096, C=4096 — LLaMA-2-7B / Mistral-7B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |   108.7 |   126.5 |   266.1 |  1.16× |  2.45× |
| proj_scale | bwd |   240.7 |   346.4 |  1489.5 |  1.44× |  6.19× |
| sinkhorn | fwd |    13.8 |    95.1 |    49.6 |  6.90× |  3.60× |
| sinkhorn | bwd |    20.3 |   230.4 |   148.2 | 11.34× |  7.29× |
| aggregate | fwd |   100.2 |   108.7 |   100.9 |  1.09× |  1.01× |
| aggregate | bwd |   195.8 |   209.8 |   255.9 |  1.07× |  1.31× |
| post | fwd |   172.2 |   216.0 |   180.1 |  1.25× |  1.05× |
| post | bwd |   355.2 |   979.7 |   458.5 |  2.76× |  1.29× |
| **pipeline** | **fwd+bwd** | ** 1206.9** | ** 2312.7** | ** 2948.7** | ** 1.92×** | ** 2.44×** |

### B=2, T=4096, C=5120 — LLaMA-2-13B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    73.4 |   150.8 |   167.9 |  2.05× |  2.29× |
| proj_scale | bwd |   153.6 |   306.5 |   873.2 |  2.00× |  5.69× |
| sinkhorn | fwd |     9.7 |    93.7 |    26.2 |  9.62× |  2.69× |
| sinkhorn | bwd |    13.3 |   207.2 |    87.1 | 15.59× |  6.56× |
| aggregate | fwd |    64.1 |    72.6 |    65.2 |  1.13× |  1.02× |
| aggregate | bwd |   123.3 |   128.5 |   160.6 |  1.04× |  1.30× |
| post | fwd |   107.5 |   138.0 |   113.3 |  1.28× |  1.05× |
| post | bwd |   222.3 |   595.2 |   286.3 |  2.68× |  1.29× |
| **pipeline** | **fwd+bwd** | **  767.3** | ** 1692.4** | ** 1779.8** | ** 2.21×** | ** 2.32×** |

### B=1, T=4096, C=6656 — LLaMA-2-33B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    53.0 |   192.4 |   111.1 |  3.63× |  2.10× |
| proj_scale | bwd |   103.9 |   161.3 |   437.7 |  1.55× |  4.21× |
| sinkhorn | fwd |     7.8 |    93.2 |    14.9 | 11.96× |  1.91× |
| sinkhorn | bwd |    11.1 |   206.2 |    84.1 | 18.61× |  7.59× |
| aggregate | fwd |    42.9 |   210.6 |    44.4 |  4.91× |  1.03× |
| aggregate | bwd |    81.5 |    95.8 |   114.7 |  1.18× |  1.41× |
| post | fwd |    71.6 |   121.6 |    80.3 |  1.70× |  1.12× |
| post | bwd |   146.4 |   438.9 |   187.8 |  3.00× |  1.28× |
| **pipeline** | **fwd+bwd** | **  518.2** | ** 1520.1** | ** 1075.0** | ** 2.93×** | ** 2.07×** |

### B=1, T=4096, C=8192 — LLaMA-2-70B / Qwen-72B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    65.7 |   235.6 |   134.0 |  3.58× |  2.04× |
| proj_scale | bwd |   125.7 |   158.9 |   439.9 |  1.26× |  3.50× |
| sinkhorn | fwd |     7.8 |    93.1 |    14.9 | 11.90× |  1.90× |
| sinkhorn | bwd |    11.0 |   221.3 |    81.7 | 20.09× |  7.42× |
| aggregate | fwd |    51.9 |    57.3 |    53.3 |  1.10× |  1.03× |
| aggregate | bwd |    99.6 |    99.4 |   133.4 |  1.00× |  1.34× |
| post | fwd |    86.6 |   117.0 |    93.8 |  1.35× |  1.08× |
| post | bwd |   179.2 |   451.2 |   226.9 |  2.52× |  1.27× |
| **pipeline** | **fwd+bwd** | **  627.4** | ** 1433.9** | ** 1177.9** | ** 2.29×** | ** 1.88×** |

### B=4, T=8192, C=4096 — LLaMA-3-8B (8k ctx, B=4)

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |   200.2 |   247.4 |   467.5 |  1.24× |  2.34× |
| proj_scale | bwd |   473.1 |   696.4 |  2984.4 |  1.47× |  6.31× |
| sinkhorn | fwd |    22.0 |   101.5 |    96.0 |  4.61× |  4.36× |
| sinkhorn | bwd |    29.1 |   265.1 |   240.3 |  9.11× |  8.26× |
| aggregate | fwd |   193.2 |   208.0 |   193.2 |  1.08× |  1.00× |
| aggregate | bwd |   388.5 |   413.4 |   505.1 |  1.06× |  1.30× |
| post | fwd |   341.7 |   427.5 |   354.6 |  1.25× |  1.04× |
| post | bwd |   704.2 |  1952.6 |   902.3 |  2.77× |  1.28× |
| **pipeline** | **fwd+bwd** | ** 2351.9** | ** 4311.9** | ** 5743.4** | ** 1.83×** | ** 2.44×** |

### B=1, T=8192, C=4096 — LLaMA-3-8B (8k ctx, B=1)

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    65.2 |   121.8 |   137.4 |  1.87× |  2.11× |
| proj_scale | bwd |   125.3 |   177.7 |   738.1 |  1.42× |  5.89× |
| sinkhorn | fwd |     9.8 |    93.6 |    26.3 |  9.59× |  2.69× |
| sinkhorn | bwd |    13.2 |   221.0 |    88.3 | 16.69× |  6.67× |
| aggregate | fwd |    52.4 |    58.5 |    52.1 |  1.12× |  0.99× |
| aggregate | bwd |    99.3 |   108.5 |   131.0 |  1.09× |  1.32× |
| post | fwd |    86.6 |   111.0 |    91.8 |  1.28× |  1.06× |
| post | bwd |   181.2 |   493.8 |   235.2 |  2.72× |  1.30× |
| **pipeline** | **fwd+bwd** | **  633.0** | ** 1385.8** | ** 1500.2** | ** 2.19×** | ** 2.37×** |

### B=4, T=8192, C=7168 — DeepSeek-V2-Lite

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |   336.8 |   421.1 |   790.3 |  1.25× |  2.35× |
| proj_scale | bwd |   820.4 |  1297.7 |  3534.8 |  1.58× |  4.31× |
| sinkhorn | fwd |    22.1 |   101.7 |    96.2 |  4.60× |  4.35× |
| sinkhorn | bwd |    29.0 |   264.5 |   235.4 |  9.14× |  8.13× |
| aggregate | fwd |   332.4 |   339.2 |   332.8 |  1.02× |  1.00× |
| aggregate | bwd |   679.7 |   649.6 |   877.6 |  0.96× |  1.29× |
| post | fwd |   600.0 |   761.7 |   615.9 |  1.27× |  1.03× |
| post | bwd |  1217.3 |  3149.8 |  1527.4 |  2.59× |  1.25× |
| **pipeline** | **fwd+bwd** | ** 4037.7** | ** 6985.2** | ** 8010.4** | ** 1.73×** | ** 1.98×** |

### B=1, T=8192, C=8192 — LLaMA-3-70B (8k ctx)

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |   108.2 |   238.0 |   260.5 |  2.20× |  2.41× |
| proj_scale | bwd |   240.0 |   309.7 |   855.8 |  1.29× |  3.57× |
| sinkhorn | fwd |     9.7 |    93.5 |    26.3 |  9.62× |  2.70× |
| sinkhorn | bwd |    13.2 |   216.4 |   100.4 | 16.36× |  7.59× |
| aggregate | fwd |    99.9 |   105.7 |   101.8 |  1.06× |  1.02× |
| aggregate | bwd |   195.6 |   190.3 |   259.6 |  0.97× |  1.33× |
| post | fwd |   171.4 |   225.5 |   182.4 |  1.32× |  1.06× |
| post | bwd |   351.1 |   895.4 |   445.9 |  2.55× |  1.27× |
| **pipeline** | **fwd+bwd** | ** 1189.2** | ** 2274.5** | ** 2232.7** | ** 1.91×** | ** 1.88×** |

### B=1, T=8192, C=16384 — LLaMA-3-405B (8k ctx)

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |   196.4 |   470.6 |   505.2 |  2.40× |  2.57× |
| proj_scale | bwd |   474.3 |   568.4 |  2051.1 |  1.20× |  4.32× |
| sinkhorn | fwd |     9.8 |    93.6 |    26.3 |  9.55× |  2.69× |
| sinkhorn | bwd |    13.2 |   239.4 |    88.4 | 18.07× |  6.67× |
| aggregate | fwd |   193.1 |   197.5 |   197.8 |  1.02× |  1.02× |
| aggregate | bwd |   387.8 |   364.9 |   511.1 |  0.94× |  1.32× |
| post | fwd |   341.6 |   452.4 |   362.5 |  1.32× |  1.06× |
| post | bwd |   704.4 |  1697.9 |   851.3 |  2.41× |  1.21× |
| **pipeline** | **fwd+bwd** | ** 2320.7** | ** 4084.6** | ** 4593.7** | ** 1.76×** | ** 1.98×** |

### B=8, T=2048, C=2560 — GPT-3-1.3B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    73.9 |    81.1 |   174.6 |  1.10× |  2.36× |
| proj_scale | bwd |   153.6 |   316.8 |  1480.4 |  2.06× |  9.64× |
| sinkhorn | fwd |    13.3 |    94.8 |    49.7 |  7.13× |  3.74× |
| sinkhorn | bwd |    20.1 |   213.9 |   149.4 | 10.65× |  7.44× |
| aggregate | fwd |    64.0 |   341.4 |    64.6 |  5.33× |  1.01× |
| aggregate | bwd |   123.8 |   168.4 |   161.0 |  1.36× |  1.30× |
| post | fwd |   107.3 |   191.0 |   115.8 |  1.78× |  1.08× |
| post | bwd |   223.3 |   752.9 |   309.3 |  3.37× |  1.39× |
| **pipeline** | **fwd+bwd** | **  779.4** | ** 2160.2** | ** 2504.7** | ** 2.77×** | ** 3.21×** |

### B=4, T=2048, C=4096 — GPT-3-6.7B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    65.3 |   121.9 |   137.3 |  1.87× |  2.10× |
| proj_scale | bwd |   125.1 |   177.9 |   739.5 |  1.42× |  5.91× |
| sinkhorn | fwd |     9.8 |    93.6 |    26.2 |  9.59× |  2.68× |
| sinkhorn | bwd |    13.3 |   221.1 |    88.3 | 16.67× |  6.66× |
| aggregate | fwd |    52.1 |    58.6 |    52.2 |  1.13× |  1.00× |
| aggregate | bwd |    99.6 |   108.6 |   130.8 |  1.09× |  1.31× |
| post | fwd |    86.8 |   110.8 |    91.9 |  1.28× |  1.06× |
| post | bwd |   181.3 |   494.0 |   235.0 |  2.73× |  1.30× |
| **pipeline** | **fwd+bwd** | **  633.1** | ** 1386.4** | ** 1501.2** | ** 2.19×** | ** 2.37×** |

### B=1, T=2048, C=12288 — GPT-3-175B

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    51.2 |   342.4 |   192.4 |  6.69× |  3.76× |
| proj_scale | bwd |    96.9 |   121.5 |   431.5 |  1.25× |  4.45× |
| sinkhorn | fwd |     7.2 |    93.0 |     8.6 | 12.98× |  1.20× |
| sinkhorn | bwd |    11.2 |   205.0 |    78.4 | 18.31× |  7.00× |
| aggregate | fwd |    40.0 |    45.1 |    45.8 |  1.13× |  1.14× |
| aggregate | bwd |    75.2 |    77.5 |   101.2 |  1.03× |  1.34× |
| post | fwd |    65.8 |    90.2 |    74.8 |  1.37× |  1.14× |
| post | bwd |   136.7 |   329.1 |   180.8 |  2.41× |  1.32× |
| **pipeline** | **fwd+bwd** | **  484.2** | ** 1303.8** | ** 1113.4** | ** 2.69×** | ** 2.30×** |

### B=3, T=8192, C=2048

| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |
|---|---|---:|---:|---:|---:|---:|
| proj_scale | fwd |    87.9 |   129.0 |   211.7 |  1.47× |  2.41× |
| proj_scale | bwd |   182.9 |   454.5 |  2159.0 |  2.49× | 11.80× |
| sinkhorn | fwd |    18.6 |   100.8 |    72.9 |  5.42× |  3.92× |
| sinkhorn | bwd |    28.3 |   237.8 |   214.5 |  8.40× |  7.58× |
| aggregate | fwd |    76.3 |   108.8 |    76.6 |  1.43× |  1.00× |
| aggregate | bwd |   148.0 |   184.0 |   189.8 |  1.24× |  1.28× |
| post | fwd |   128.8 |   190.5 |   130.0 |  1.48× |  1.01× |
| post | bwd |   268.0 |   881.9 |   383.9 |  3.29× |  1.43× |
| **pipeline** | **fwd+bwd** | **  938.8** | ** 2287.4** | ** 3438.5** | ** 2.44×** | ** 3.66×** |

## Geomean: how many times faster triton is

Geometric mean of per-config speedup ratios. `3.00×` means triton is on average 3× faster than that framework on that op. `<1×` would mean triton is slower; there are no such cases below.

| op | dir | tr vs cu (geomean) | tr vs tl (geomean) |
|---|---|---:|---:|
| proj_scale | fwd |  2.02× |  2.38× |
| proj_scale | bwd |  1.54× |  5.46× |
| sinkhorn | fwd |  8.27× |  2.79× |
| sinkhorn | bwd | 13.93× |  7.28× |
| aggregate | fwd |  1.41× |  1.02× |
| aggregate | bwd |  1.07× |  1.32× |
| post | fwd |  1.37× |  1.06× |
| post | bwd |  2.74× |  1.30× |

## Whole-pipeline: how many times faster triton is

Sum of all (op, fwd+bwd) times per config, then ratio, then geomean across configs.

- triton vs cutile (full pipeline): **2.19× faster**
- triton vs tilelang (full pipeline): **2.33× faster**
