"""
path-D resampling analysis: load the cached npz and build subsampling curves (pure NumPy, seconds).

Main curve: direction recomputed per subset (no leakage; simulates the real "only n cells" situation).
Control curve: fixed-direction ablation (direction from the full pool; isolates "magnitude attenuation
  vs direction error").

Output: mean recovered Pearson r per level + 95% CI; CCND1 (strong signal) vs TIMP1 (weak-signal control) plot.

Usage: python resample_pathd.py   (auto-scans pathd_cache/*_cache.npz)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("resample")

CACHE_DIR = Path("benchmark_output/pathd_cache")
LEVELS = [2, 5, 10, 20, 40, 80, 160, 320]   # "full" appended automatically
SEED = 0


def reps_for(n):
    """Repeat-count schedule (small n has high variance and needs more repeats; full is deterministic = 1)."""
    if n <= 20:
        return 100
    elif n <= 80:
        return 50
    else:
        return 20


def load_cache(npz_path):
    """Load the cache, rebuild dense cos_mat / proj_mat [n_valid, n_genes] (NaN-filled) + ground truth."""
    d = np.load(npz_path, allow_pickle=False)
    gene = d["gene"]; cellpos = d["cellpos"]; cos = d["cos"]; proj = d["proj"]
    n_valid = int(d["n_valid"])
    gt_tokens = d["gt_tokens"]; gt_fc = d["gt_fc"]
    axis_check = float(d["axis_check"])
    gene_name = str(d["gene_name"][0])

    # gene token -> column index
    uniq = np.unique(gene)
    tok2col = {t: i for i, t in enumerate(uniq)}
    col = np.array([tok2col[t] for t in gene])

    cos_mat = np.full((n_valid, len(uniq)), np.nan, dtype=np.float32)
    proj_mat = np.full((n_valid, len(uniq)), np.nan, dtype=np.float32)
    cos_mat[cellpos, col] = cos
    proj_mat[cellpos, col] = proj

    # align ground truth to columns
    gt_map = {int(t): float(f) for t, f in zip(gt_tokens, gt_fc)}
    gt_vec = np.array([gt_map.get(int(t), np.nan) for t in uniq], dtype=np.float64)

    return {
        "name": gene_name, "n_valid": n_valid, "axis_check": axis_check,
        "cos_mat": cos_mat, "proj_mat": proj_mat, "uniq": uniq, "gt": gt_vec,
    }


def r_for_subset(cos_mat, proj_mat, gt, rows, fixed_dir=None):
    """For a cell subset `rows`, compute the Pearson r between signed_shift and gt.
    fixed_dir: None = direction recomputed per subset; otherwise use the supplied full-pool direction signs."""
    sub_cos = cos_mat[rows, :]
    sub_proj = proj_mat[rows, :]
    with np.errstate(invalid="ignore"):
        magnitude = np.nanmean(1.0 - sub_cos, axis=0)          # magnitude
        if fixed_dir is None:
            direction = np.sign(np.nanmean(sub_proj, axis=0))   # direction per subset
        else:
            direction = fixed_dir
    signed = magnitude * direction
    # valid genes: signed not NaN and gt not NaN
    mask = ~np.isnan(signed) & ~np.isnan(gt)
    if mask.sum() < 10:
        return np.nan, int(mask.sum())
    r, _ = pearsonr(signed[mask], gt[mask])
    return r, int(mask.sum())


def run_curve(cache, fixed_direction=False):
    """Run one subsampling curve; return a DataFrame (level, r_mean, ci_lo, ci_hi, n_genes)."""
    cos_mat, proj_mat, gt = cache["cos_mat"], cache["proj_mat"], cache["gt"]
    n_valid = cache["n_valid"]
    rng = np.random.default_rng(SEED)

    # full-pool direction (used by the fixed_direction ablation)
    with np.errstate(invalid="ignore"):
        full_dir = np.sign(np.nanmean(proj_mat, axis=0))

    levels = [L for L in LEVELS if L < n_valid] + [n_valid]
    rows_out = []
    for L in levels:
        nrep = 1 if L == n_valid else reps_for(L)
        rs, ng = [], []
        for _ in range(nrep):
            rows = rng.choice(n_valid, size=L, replace=False) if L < n_valid else np.arange(n_valid)
            r, n = r_for_subset(cos_mat, proj_mat, gt, rows,
                                fixed_dir=full_dir if fixed_direction else None)
            if not np.isnan(r):
                rs.append(r); ng.append(n)
        rs = np.array(rs)
        rows_out.append({
            "level": L, "n_reps": len(rs),
            "r_mean": float(np.mean(rs)) if len(rs) else np.nan,
            "ci_lo": float(np.percentile(rs, 2.5)) if len(rs) > 1 else (float(rs[0]) if len(rs) else np.nan),
            "ci_hi": float(np.percentile(rs, 97.5)) if len(rs) > 1 else (float(rs[0]) if len(rs) else np.nan),
            "n_genes": int(np.median(ng)) if ng else 0,
        })
    return pd.DataFrame(rows_out)


def main():
    caches = sorted(CACHE_DIR.glob("*_cache.npz"))
    if not caches:
        log.error(f"No cache files found in {CACHE_DIR}")
        return
    log.info(f"Found {len(caches)} caches: {[c.name for c in caches]}")

    results = {}
    for npz in caches:
        cache = load_cache(npz)
        log.info(f"\n{'='*60}")
        log.info(f"Gene {cache['name']}: valid pool {cache['n_valid']}, axis stability cos={cache['axis_check']:.5f}")

        main_curve = run_curve(cache, fixed_direction=False)
        abl_curve = run_curve(cache, fixed_direction=True)

        log.info("--- main curve (direction recomputed per subset) ---")
        print(main_curve.to_string(index=False))
        log.info("--- control (fixed-direction ablation) ---")
        print(abl_curve.to_string(index=False))

        results[cache["name"]] = {"main": main_curve, "abl": abl_curve, "cache": cache}
        main_curve.to_csv(CACHE_DIR / f"{cache['name']}_curve_main.csv", index=False)
        abl_curve.to_csv(CACHE_DIR / f"{cache['name']}_curve_abl.csv", index=False)

    # -- plot ---------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"CCND1": "C0", "TIMP1": "C1"}
        for name, res in results.items():
            c = colors.get(name, None)
            m = res["main"]
            ax.plot(m["level"], m["r_mean"], "-o", color=c, label=f"{name} (direction per subset)")
            ax.fill_between(m["level"], m["ci_lo"], m["ci_hi"], color=c, alpha=0.2)
            a = res["abl"]
            ax.plot(a["level"], a["r_mean"], "--s", color=c, alpha=0.6, markersize=4,
                    label=f"{name} (fixed-direction ablation)")
        ax.set_xscale("log")
        ax.set_xlabel("number of valid cells (n valid cells, log scale)")
        ax.set_ylabel("Recovered Pearson r (predicted vs measured log2FC)")
        ax.set_title("path-D subsampling: coverage (n valid cells) -> recovered estimate")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = CACHE_DIR / "downsampling_curve.png"
        fig.tight_layout(); fig.savefig(out, dpi=150)
        log.info(f"\nFigure saved: {out}")
    except Exception as e:
        log.warning(f"Plotting failed (does not affect data): {e}")


if __name__ == "__main__":
    main()
