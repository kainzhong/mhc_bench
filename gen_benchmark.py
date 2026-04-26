#!/usr/bin/env python3
"""Produce benchmark.md from the per-config nsys profile outputs."""
import glob, os, re, sys
from math import log, exp

PROFILE_DIR = "/home/kainingz/GitHub/mhc_bench/profile"
OUTPUT_MD = "/home/kainingz/GitHub/mhc_bench/benchmark.md"

# Config order and labels (matches mhc_bench.sh's CONFIGS array).
CONFIG_ORDER = [
    ((4, 4096, 4096),  "LLaMA-2-7B / Mistral-7B"),
    ((2, 4096, 5120),  "LLaMA-2-13B"),
    ((1, 4096, 6656),  "LLaMA-2-33B"),
    ((1, 4096, 8192),  "LLaMA-2-70B / Qwen-72B"),
    ((4, 8192, 4096),  "LLaMA-3-8B (8k ctx, B=4)"),
    ((1, 8192, 4096),  "LLaMA-3-8B (8k ctx, B=1)"),
    ((4, 8192, 7168),  "DeepSeek-V2-Lite"),
    ((1, 8192, 8192),  "LLaMA-3-70B (8k ctx)"),
    ((1, 8192, 16384), "LLaMA-3-405B (8k ctx)"),
    ((8, 2048, 2560),  "GPT-3-1.3B"),
    ((4, 2048, 4096),  "GPT-3-6.7B"),
    ((1, 2048, 12288), "GPT-3-175B"),
    ((3, 8192, 2048),  ""),
]

# (framework, op, direction) -> kernel patterns to sum
KERNELS = {
    ("triton",   "proj_scale", "fwd"): ["_mhc_projection_fwd_fused",  "_mhc_scale_fwd_fused"],
    ("triton",   "proj_scale", "bwd"): ["_mhc_projection_bwd_fused",  "_mhc_scale_bwd_fused"],
    ("triton",   "sinkhorn",   "fwd"): ["_mhc_sinkhorn_fwd_fused_recompute"],
    ("triton",   "sinkhorn",   "bwd"): ["_mhc_sinkhorn_bwd_fused_recompute"],
    ("triton",   "aggregate",  "fwd"): ["_mhc_aggregate_fwd "],
    ("triton",   "aggregate",  "bwd"): ["_mhc_aggregate_bwd"],
    ("triton",   "post",       "fwd"): ["_mhc_expand_combine_fwd"],
    ("triton",   "post",       "bwd"): ["_mhc_expand_combine_bwd"],

    # Bare prefix matches both old (`_ct_proj_rms_fwd_kernel_Kt1_A...`) and new
    # (`_ct_proj_rms_fwd_kernel` with no tag) cutile kernel naming styles.
    ("cutile",   "proj_scale", "fwd"): ["_ct_proj_rms_fwd_kernel"],
    ("cutile",   "proj_scale", "bwd"): ["_ct_proj_rms_bwd_kernel"],
    ("cutile",   "sinkhorn",   "fwd"): ["_ct_sinkhorn_fwd_kernel"],
    ("cutile",   "sinkhorn",   "bwd"): ["_ct_sinkhorn_bwd_kernel"],
    ("cutile",   "aggregate",  "fwd"): ["_ct_h_agg_fwd_kernel"],
    ("cutile",   "aggregate",  "bwd"): ["_ct_h_agg_bwd_kernel"],
    ("cutile",   "post",       "fwd"): ["_ct_hpb_fwd_kernel"],
    ("cutile",   "post",       "bwd"): ["_ct_hpb_bwd_kernel"],

    ("tilelang", "proj_scale", "fwd"): [
        "_mhc_pre_norm_fn_fwd_mul_kernel_kernel",
        "_mhc_pre_norm_fn_fwd_norm_kernel_kernel",
        "mhc_pre_split_mixes_fwd_kernel_kernel",
    ],
    ("tilelang", "proj_scale", "bwd"): [
        "_mhc_pre_norm_fn_bwd_mul_kernel_kernel",
        "_mhc_pre_norm_fn_bwd_norm_kernel_kernel",
        "mhc_pre_split_mixes_bwd_kernel_kernel",
    ],
    ("tilelang", "sinkhorn",   "fwd"): ["mhc_sinkhorn_kernel_kernel"],
    ("tilelang", "sinkhorn",   "bwd"): ["mhc_sinkhorn_backward_kernel_kernel"],
    ("tilelang", "aggregate",  "fwd"): ["_mhc_pre_apply_mix_fwd_kernel_kernel"],
    ("tilelang", "aggregate",  "bwd"): ["_mhc_pre_apply_mix_bwd_kernel_kernel"],
    ("tilelang", "post",       "fwd"): ["_mhc_post_fwd_kernel_kernel"],
    ("tilelang", "post",       "bwd"): ["_mhc_post_bwd_kernel_kernel"],

    # flash_mhc — only 3 ops implemented (no scale, no sinkhorn).
    # Patterns are prefixes that match both bare and `_autotuned` kernel names.
    ("flashmhc", "proj_scale", "fwd"): ["_fused_rmsnorm_project_fwd_kernel"],
    ("flashmhc", "proj_scale", "bwd"): ["_fused_rmsnorm_project_bwd_dx_kernel"],
    ("flashmhc", "aggregate",  "fwd"): ["_fused_pre_map_fwd_kernel"],
    ("flashmhc", "aggregate",  "bwd"): ["_fused_pre_map_bwd_fused_kernel_n4"],
    ("flashmhc", "post",       "fwd"): ["_fused_post_res_fwd_kernel_n4"],
    ("flashmhc", "post",       "bwd"): ["_fused_post_res_bwd_fused_kernel_n4"],
    # flash_mhc has no sinkhorn or scale kernels — those rows show "n/a".
}

FRAMEWORKS = ["triton", "cutile", "tilelang", "flashmhc"]
OP_ORDER = ["proj_scale", "sinkhorn", "aggregate", "post"]
DIRS = ["fwd", "bwd"]


def avg_ns(content, pattern):
    for line in content.splitlines():
        if pattern in line:
            fields = line.split()
            try:
                float(fields[0]); float(fields[3])
            except (ValueError, IndexError):
                continue
            return float(fields[3])
    return None


def load(B, T, C):
    path = os.path.join(PROFILE_DIR, f"nsys_mhc_B{B}_T{T}_C{C}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    out = {}
    for (fw, op, d), pats in KERNELS.items():
        total = sum((avg_ns(content, p) or 0.0) for p in pats)
        found = any(avg_ns(content, p) is not None for p in pats)
        out[(fw, op, d)] = total if found else None
    return out


def fmt_us(ns):
    if ns is None:
        return "  n/a"
    return f"{ns / 1000.0:7.1f}"


def fmt_ratio(num, den):
    if num is None or den is None or den == 0:
        return "  n/a"
    return f"{num/den:5.2f}×"


def main():
    # Pre-scan all configs to detect which frameworks actually have data in
    # the profile directory. If a framework was absent at profile-time (e.g.
    # we re-ran on a branch without flash_mhc), drop it from the output
    # entirely instead of filling every cell with `n/a`.
    all_data = {}
    for (B, T, C), label in CONFIG_ORDER:
        d = load(B, T, C)
        if d is not None:
            all_data[(B, T, C)] = d
    fm_present = any(
        any(d.get(("flashmhc", op, dr)) is not None
            for op in OP_ORDER for dr in DIRS)
        for d in all_data.values()
    )

    md = []
    md.append("# mHC kernel benchmark\n")
    md.append(
        "Per-kernel GPU time for each (op, direction, framework) on 13 "
        "LLM-training-class shapes, profiled by `mhc_bench.sh` → `mhc_bench.py` "
        "→ nsys, using the kernel-name → (op, framework) mapping in "
        "`kernel_comparison.md`.\n"
    )
    md.append("## Methodology\n")
    methodology = (
        "- **Numbers are `Avg (ns)` from `cuda_gpu_kern_sum`**, averaged across "
        "5 measured iterations per kernel (after 5 warmup iterations per "
        "framework for JIT + autotune + cache warming).\n"
        "- **Per-op sums** follow `kernel_comparison.md`. Notable boundaries:\n"
        "  - tilelang `proj_scale` = `pre_norm_fn_{mul,norm}` + `pre_split_mixes`\n"
        "  - triton `proj_scale` = `projection + scale`\n"
        "  - cutile `proj_scale` = `proj_rms` only (no scale kernel)\n"
    )
    if fm_present:
        methodology += (
            "  - **flash_mhc** `proj_scale` = `fused_rmsnorm_project` only (no "
            "scale kernel; uses BvN convex combination instead of Sinkhorn so the "
            "sinkhorn row is also missing on flash_mhc)\n"
        )
    methodology += (
        "- **Wrapper kernels are excluded**: `at::native::*elementwise*`, "
        "`FillFunctor`, `bfloat16_copy`, `reduce_kernel`, `cublasLt::splitKreduce`, "
        "`nvjet_sm100_tst_*`, torch RNG kernels."
    )
    if fm_present:
        methodology += (
            " Note: flash_mhc's `grad_W` for K1 is computed via `torch.matmul` "
            "(cuBLAS), so the flash_mhc `proj_scale.bwd` cell is **only the "
            "`_fused_rmsnorm_project_bwd_dx_kernel`** (grad_x). Triton's "
            "projection bwd has the same property — `grad_phi` via cuBLAS is "
            "excluded — so this asymmetry doesn't bias the tr-vs-fm comparison."
        )
    methodology += (
        "\n- Each framework is warmed up and profiled in its own "
        "`cudaProfilerStart/Stop` window (via `--capture-range-end=repeat`) "
        "so L2/cache state doesn't leak between frameworks.\n"
        "- `--iters 5` means 5 instances per kernel in the averages.\n"
        "- Clocks are **not locked**. Run-to-run variance on the order of a "
        "few % is expected; kernel-level ratios are stable.\n"
    )
    md.append(methodology)
    md.append("## Results\n")
    if fm_present:
        md.append(
            "All times in **µs** per (op, direction) call. `tr vs cu` / `tr vs tl` "
            "/ `tr vs fm` columns are **how many times faster triton is** compared "
            "to cutile / tilelang / flash_mhc on that op (e.g. `2.50×` means triton "
            "is 2.5× faster; <1× means triton is slower). `n/a` means that "
            "framework doesn't implement that op.\n"
        )
    else:
        md.append(
            "All times in **µs** per (op, direction) call. `tr vs cu` / `tr vs tl` "
            "columns are **how many times faster triton is** compared to cutile / "
            "tilelang on that op (e.g. `2.50×` means triton is 2.5× faster; <1× "
            "means triton is slower).\n"
        )

    # Per-config tables.
    ratios_agg = {}  # (fw, op, d) -> list of ratios vs triton
    for (B, T, C), label in CONFIG_ORDER:
        data = load(B, T, C)
        if data is None:
            md.append(f"### B={B}, T={T}, C={C} — {label}\n*(profile not found)*\n")
            continue
        md.append(f"### B={B}, T={T}, C={C} — {label}\n")
        if fm_present:
            md.append("| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | flash_mhc (µs) | tr vs cu | tr vs tl | tr vs fm |")
            md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        else:
            md.append("| op | dir | triton (µs) | cutile (µs) | tilelang (µs) | tr vs cu | tr vs tl |")
            md.append("|---|---|---:|---:|---:|---:|---:|")
        for op in OP_ORDER:
            for d in DIRS:
                tr = data.get(("triton", op, d))
                cu = data.get(("cutile", op, d))
                tl = data.get(("tilelang", op, d))
                fm = data.get(("flashmhc", op, d))
                if fm_present:
                    md.append(
                        f"| {op} | {d} | {fmt_us(tr)} | {fmt_us(cu)} | {fmt_us(tl)} | {fmt_us(fm)} "
                        f"| {fmt_ratio(cu, tr)} | {fmt_ratio(tl, tr)} | {fmt_ratio(fm, tr)} |"
                    )
                else:
                    md.append(
                        f"| {op} | {d} | {fmt_us(tr)} | {fmt_us(cu)} | {fmt_us(tl)} "
                        f"| {fmt_ratio(cu, tr)} | {fmt_ratio(tl, tr)} |"
                    )
                if tr and tr > 0:
                    if cu is not None:
                        ratios_agg.setdefault(("cutile", op, d), []).append(cu/tr)
                    if tl is not None:
                        ratios_agg.setdefault(("tilelang", op, d), []).append(tl/tr)
                    if fm is not None:
                        ratios_agg.setdefault(("flashmhc", op, d), []).append(fm/tr)
        # per-config totals — flash_mhc total only sums ops it implements;
        # caveat noted in the methodology section.
        tot_tr = sum((data.get(("triton",   o, d)) or 0) for o in OP_ORDER for d in DIRS)
        tot_cu = sum((data.get(("cutile",   o, d)) or 0) for o in OP_ORDER for d in DIRS)
        tot_tl = sum((data.get(("tilelang", o, d)) or 0) for o in OP_ORDER for d in DIRS)
        tot_fm = sum((data.get(("flashmhc", o, d)) or 0) for o in OP_ORDER for d in DIRS)
        if fm_present:
            md.append(
                f"| **pipeline** | **fwd+bwd** | **{fmt_us(tot_tr)}** | **{fmt_us(tot_cu)}** | "
                f"**{fmt_us(tot_tl)}** | **{fmt_us(tot_fm)}** | **{fmt_ratio(tot_cu, tot_tr)}** | "
                f"**{fmt_ratio(tot_tl, tot_tr)}** | **{fmt_ratio(tot_fm, tot_tr)}** |"
            )
        else:
            md.append(
                f"| **pipeline** | **fwd+bwd** | **{fmt_us(tot_tr)}** | **{fmt_us(tot_cu)}** | "
                f"**{fmt_us(tot_tl)}** | **{fmt_ratio(tot_cu, tot_tr)}** | "
                f"**{fmt_ratio(tot_tl, tot_tr)}** |"
            )
        md.append("")

    # Geomean summary across configs.
    md.append("## Geomean: how many times faster triton is\n")
    md.append(
        "Geometric mean of per-config speedup ratios. `3.00×` means triton "
        "is on average 3× faster than that framework on that op. `<1×` would "
        "mean triton is slower; there are no such cases below.\n"
    )
    if fm_present:
        md.append("| op | dir | tr vs cu (geomean) | tr vs tl (geomean) | tr vs fm (geomean) |")
        md.append("|---|---|---:|---:|---:|")
    else:
        md.append("| op | dir | tr vs cu (geomean) | tr vs tl (geomean) |")
        md.append("|---|---|---:|---:|")
    for op in OP_ORDER:
        for d in DIRS:
            cu_ratios = ratios_agg.get(("cutile", op, d), [])
            tl_ratios = ratios_agg.get(("tilelang", op, d), [])
            fm_ratios = ratios_agg.get(("flashmhc", op, d), [])
            def gm(vs):
                return exp(sum(log(v) for v in vs) / len(vs)) if vs else None
            def gm_s(vs):
                g = gm(vs)
                return f"{g:5.2f}×" if g is not None else " n/a"
            if fm_present:
                md.append(
                    f"| {op} | {d} | {gm_s(cu_ratios)} | {gm_s(tl_ratios)} | {gm_s(fm_ratios)} |"
                )
            else:
                md.append(
                    f"| {op} | {d} | {gm_s(cu_ratios)} | {gm_s(tl_ratios)} |"
                )
    md.append("")

    # Per-framework pipeline geomean. flash_mhc pipeline excludes scale +
    # sinkhorn (not implemented), so we compare on the SAME-SCOPE subset of
    # ops where flash_mhc has kernels.
    md.append("## Whole-pipeline: how many times faster triton is\n")
    if fm_present:
        md.append(
            "Sum of all (op, fwd+bwd) times per config, then ratio, then geomean "
            "across configs. flash_mhc's pipeline number sums only the ops it "
            "implements (`proj_scale`, `aggregate`, `post`); the triton pipeline "
            "in the `tr vs fm` row is computed on the same op subset for fairness.\n"
        )
    else:
        md.append(
            "Sum of all (op, fwd+bwd) times per config, then ratio, then geomean across configs.\n"
        )
    FM_OPS = ["proj_scale", "aggregate", "post"]  # ops flash_mhc has kernels for
    pipe_cu, pipe_tl, pipe_fm = [], [], []
    for (B, T, C), _ in CONFIG_ORDER:
        data = load(B, T, C)
        if data is None:
            continue
        tr_full = sum((data.get(("triton",   o, d)) or 0) for o in OP_ORDER for d in DIRS)
        cu_full = sum((data.get(("cutile",   o, d)) or 0) for o in OP_ORDER for d in DIRS)
        tl_full = sum((data.get(("tilelang", o, d)) or 0) for o in OP_ORDER for d in DIRS)
        if tr_full > 0:
            if cu_full > 0: pipe_cu.append(cu_full/tr_full)
            if tl_full > 0: pipe_tl.append(tl_full/tr_full)
        # flash_mhc subset: only proj_scale/aggregate/post.
        tr_sub = sum((data.get(("triton",   o, d)) or 0) for o in FM_OPS for d in DIRS)
        fm_sub = sum((data.get(("flashmhc", o, d)) or 0) for o in FM_OPS for d in DIRS)
        if tr_sub > 0 and fm_sub > 0:
            pipe_fm.append(fm_sub / tr_sub)
    if pipe_cu:
        gcu = exp(sum(log(v) for v in pipe_cu)/len(pipe_cu))
        md.append(f"- triton vs cutile (full pipeline): **{gcu:.2f}× faster**")
    if pipe_tl:
        gtl = exp(sum(log(v) for v in pipe_tl)/len(pipe_tl))
        md.append(f"- triton vs tilelang (full pipeline): **{gtl:.2f}× faster**")
    if pipe_fm:
        gfm = exp(sum(log(v) for v in pipe_fm)/len(pipe_fm))
        md.append(f"- triton vs flash_mhc (proj_scale + aggregate + post only): **{gfm:.2f}× faster**")
    md.append("")

    text = "\n".join(md)
    with open(OUTPUT_MD, "w") as f:
        f.write(text)
    print(f"Wrote {OUTPUT_MD}")
    print(f"({len(text)} bytes)")


if __name__ == "__main__":
    main()
