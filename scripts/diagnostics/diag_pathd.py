"""
path-D diagnostic decomposition: locate the source of CCND1's full-pool r = -0.12 (the A/B/C framework).

A: the direction heuristic is flipped (wrong expression-axis semantics, overall sign flip) -> fixable by flipping the rule
B: the cosine metric is unrelated to log2FC (embedding-change magnitude != expression-change magnitude) -> change the metric
C: the model is genuinely anti-correlated for CCND1 -> model limitation

Discrimination logic:
  direction accuracy << 50%                    -> direction systematically flipped (A or C)
  direction accuracy ~ 50%                     -> direction uninformative (points to B)
  oracle-direction upper-bound r high positive -> magnitude useful, only direction bad = A (fixable)
  oracle-direction upper-bound r still low     -> magnitude also fails = B (metric problem)
  stronger-effect genes more/less accurate     -> distinguishes A (real structure flipped) vs C

Note: globally negating signed_shift is the identity Pearson(-x,y) = -Pearson(x,y); it discriminates nothing, so it is not used.
"""

import sys, logging
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("diag")

CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")


def main():
    d = np.load(CACHE, allow_pickle=False)
    gene, cellpos, cos, proj = d["gene"], d["cellpos"], d["cos"], d["proj"]
    n_valid = int(d["n_valid"])
    gt_map = {int(t): float(f) for t, f in zip(d["gt_tokens"], d["gt_fc"])}
    name = str(d["gene_name"][0])

    uniq = np.unique(gene)
    tok2col = {t: i for i, t in enumerate(uniq)}
    col = np.array([tok2col[t] for t in gene])
    cos_mat = np.full((n_valid, len(uniq)), np.nan, np.float32)
    proj_mat = np.full((n_valid, len(uniq)), np.nan, np.float32)
    cos_mat[cellpos, col] = cos
    proj_mat[cellpos, col] = proj

    with np.errstate(invalid="ignore"):
        mag = np.nanmean(1.0 - cos_mat, axis=0)            # magnitude > 0
        dir_proj = np.nanmean(proj_mat, axis=0)            # mean direction projection
    direction = np.sign(dir_proj)                          # +/-1
    gt = np.array([gt_map.get(int(t), np.nan) for t in uniq])

    # valid genes: mag and gt not NaN, gt != 0
    m = ~np.isnan(mag) & ~np.isnan(gt) & (gt != 0)
    mag, direction, gt, dir_proj = mag[m], direction[m], gt[m], dir_proj[m]
    ss = mag * direction
    log.info(f"Gene {name}: valid genes {m.sum()}")

    # -- baseline -----------------------------------------------
    r_base, _ = pearsonr(ss, gt)
    s_base, _ = spearmanr(ss, gt)
    log.info("\n===== baseline =====")
    log.info(f"Pearson(signed_shift, log2fc)  = {r_base:+.4f}")
    log.info(f"Spearman(signed_shift, log2fc) = {s_base:+.4f}")

    # -- diagnostic 1: direction accuracy -----------------------
    log.info("\n===== diagnostic 1: direction accuracy (sign(direction)==sign(log2fc)) =====")
    for thr in [0.0, 0.1, 0.3, 0.5, 1.0]:
        sub = np.abs(gt) > thr
        if sub.sum() < 10:
            continue
        acc = np.mean(direction[sub] == np.sign(gt[sub]))
        log.info(f"  |log2fc|>{thr:<4}: direction accuracy = {acc*100:5.1f}%  (n={sub.sum()})")

    # -- diagnostic 2: does magnitude track the true effect size? --
    log.info("\n===== diagnostic 2: magnitude(1-cosine) vs true effect =====")
    r_mag_abs, _ = pearsonr(mag, np.abs(gt))
    s_mag_abs, _ = spearmanr(mag, np.abs(gt))
    r_mag_sgn, _ = pearsonr(mag, gt)
    log.info(f"  Pearson(mag, |log2fc|)  = {r_mag_abs:+.4f}  (>0 = magnitude tracks effect size)")
    log.info(f"  Spearman(mag, |log2fc|) = {s_mag_abs:+.4f}")
    log.info(f"  Pearson(mag, log2fc)    = {r_mag_sgn:+.4f}  (signed, should be ~0)")

    # -- diagnostic 3: oracle direction upper bound (key A vs B test) --
    log.info("\n===== diagnostic 3: oracle direction upper bound (direction replaced by true sign(log2fc)) =====")
    ss_oracle = mag * np.sign(gt)
    r_oracle, _ = pearsonr(ss_oracle, gt)
    s_oracle, _ = spearmanr(ss_oracle, gt)
    log.info(f"  Pearson(mag*sign(log2fc), log2fc)  = {r_oracle:+.4f}")
    log.info(f"  Spearman = {s_oracle:+.4f}")
    log.info(f"  -> if high positive: magnitude useful, only direction bad (A, fixable); if still low: magnitude also fails (B, metric problem)")

    # -- diagnostic 4: after flipping the direction rule (meaningful flip) --
    log.info("\n===== diagnostic 4: flip the direction rule (proj>0 -> down) =====")
    r_flip, _ = pearsonr(mag * (-direction), gt)
    acc_flip = np.mean((-direction) == np.sign(gt))
    log.info(f"  after flipping: Pearson = {r_flip:+.4f}, direction accuracy = {acc_flip*100:.1f}%")

    # -- diagnostic 5: up/down-regulated gene groups ------------
    log.info("\n===== diagnostic 5: true up/down-regulated genes, predicted-direction distribution =====")
    up, dn = gt > 0, gt < 0
    log.info(f"  true up-regulated genes (n={up.sum()}): model says up {np.mean(direction[up]>0)*100:.1f}% / down {np.mean(direction[up]<0)*100:.1f}%")
    log.info(f"  true down-regulated genes (n={dn.sum()}): model says up {np.mean(direction[dn]>0)*100:.1f}% / down {np.mean(direction[dn]<0)*100:.1f}%")

    # -- overall verdict ----------------------------------------
    log.info("\n===== overall verdict =====")
    acc_all = np.mean(direction == np.sign(gt))
    if r_oracle > 0.25 and acc_all < 0.45:
        verdict = "A: magnitude useful but direction systematically flipped -> flipping the rule turns the negative correlation positive (fixable)"
    elif r_oracle < 0.15:
        verdict = "B: even with perfect direction r is low -> the cosine metric is inconsistent with log2FC (metric problem, need a different quantity)"
    elif acc_all > 0.45 and acc_all < 0.55:
        verdict = "B/C: direction near random, the negative correlation comes from magnitude structure -> leans metric problem"
    else:
        verdict = "C/mixed: direction partly flipped but the oracle upper bound is mediocre -> model limitation + metric both contribute"
    log.info(f"  global direction accuracy={acc_all*100:.1f}%, oracle upper bound r={r_oracle:+.3f}")
    log.info(f"  -> {verdict}")


if __name__ == "__main__":
    main()
