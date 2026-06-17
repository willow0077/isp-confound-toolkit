"""
Readout-ladder probe: under one gene-held-out CV split, compare the generalization r of different readouts.

Question (within-perturbation, held-out gene generalization):
  why is the hand-crafted readout (mag x sign(proj)) at -0.12? how high can a learned readout get?
  do rich features add value?

Ladder (all held-out, apples-to-apples on the same folds):
  (1) hand-crafted signed_shift = mag x sign(proj)   (no params, reported directly, expected ~-0.12)
  (2) proj alone                                     (no params)
  (3) 2-feature ridge [mag, proj]                    (CV held-out, vs in-sample +0.27)
  (4) rich ridge   [mean_delta(512), mag]            (CV held-out, rich-feature ceiling)
  (5) rich+context [mean_delta, mean_wt, mag]

Note: within-perturbation r shows "delta linearly encodes log2fc within one perturbation"; it is not
cross-perturbation generalization (the deployment criterion needs a larger cache with cross-perturbation hold-out).
A weak-signal gene (TIMP1) should serve as a null: its rich r should be low; if it is also high, the probe is fitting noise.

Usage: python probe_pathd.py [GENE]   (default CCND1)
"""
import sys, logging
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("probe")

GENE = sys.argv[1] if len(sys.argv) > 1 else "CCND1"
CACHE = Path(f"benchmark_output/pathd_cache/{GENE}_cache.npz")
N_SPLITS = 5
SEED = 0


def cv_heldout_r(X, y, n_splits=N_SPLITS, seed=SEED):
    """K-fold, collect out-of-fold predictions, pool them and compute Pearson r. ridge + scaler fit on the train fold only."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    alphas = np.logspace(-2, 4, 13)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = RidgeCV(alphas=alphas).fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
    r, _ = pearsonr(oof, y)
    return r


def main():
    if not CACHE.exists():
        log.error(f"cache not found: {CACHE} (may still be running)")
        return
    d = np.load(CACHE, allow_pickle=True)
    if "mean_delta" not in d:
        log.error("this cache has no rich features (old cache); re-run with the current cache_pathd.py")
        return

    # -- rebuild per-gene mag, proj -----------------------------
    gene, cellpos, cos, proj = d["gene"], d["cellpos"], d["cos"], d["proj"]
    n = int(d["n_valid"])
    gt = {int(t): float(f) for t, f in zip(d["gt_tokens"], d["gt_fc"])}
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); P = np.full((n, len(uniq)), np.nan, np.float32)
    C[cellpos, col] = cos; P[cellpos, col] = proj
    import warnings; warnings.filterwarnings("ignore")
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    proj_by = {int(t): p for t, p in zip(uniq, np.nanmean(P, 0))}

    # -- rich features ------------------------------------------
    rich_tokens = d["rich_tokens"]; mean_delta = d["mean_delta"]; mean_wt = d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}

    # -- common gene set (mag/proj/rich/gt all present, gt != 0) --
    common = [int(t) for t in uniq
              if int(t) in gt and gt[int(t)] != 0 and int(t) in rd]
    common = np.array(common)
    y = np.array([gt[int(t)] for t in common])
    mag = np.array([mag_by[int(t)] for t in common])
    pj = np.array([proj_by[int(t)] for t in common])
    di = np.array([rd[int(t)] for t in common])
    D = mean_delta[di]      # [G,512]
    W = mean_wt[di]         # [G,512]
    log.info(f"Gene {GENE}: common genes {len(common)}, delta dim {D.shape[1]}")

    # -- ladder -------------------------------------------------
    log.info("\n===== readout ladder (held-out CV, N_SPLITS=%d) =====" % N_SPLITS)
    r_hand, _ = pearsonr(mag * np.sign(pj), y)
    r_proj, _ = pearsonr(pj, y)
    log.info(f"(1) hand-crafted mag*sign(proj) : {r_hand:+.4f}  (no params)")
    log.info(f"(2) proj alone                  : {r_proj:+.4f}  (no params)")

    # in-sample 2-feature reference
    X2 = np.column_stack([mag, pj])
    from numpy.linalg import lstsq
    Xa = np.column_stack([X2, np.ones(len(y))])
    b, *_ = lstsq(Xa, y, rcond=None)
    r2_in, _ = pearsonr(Xa @ b, y)

    r2_cv = cv_heldout_r(X2, y)
    rd_cv = cv_heldout_r(np.column_stack([D, mag]), y)
    rdw_cv = cv_heldout_r(np.column_stack([D, W, mag]), y)

    log.info(f"(3) 2-feature [mag,proj]        : held-out {r2_cv:+.4f}  (in-sample {r2_in:+.4f})")
    log.info(f"(4) rich [delta(512),mag]       : held-out {rd_cv:+.4f}")
    log.info(f"(5) rich+context [delta,WT,mag] : held-out {rdw_cv:+.4f}")

    # -- interpretation -----------------------------------------
    log.info("\n===== interpretation =====")
    best_learned = max(r2_cv, rd_cv, rdw_cv)
    log.info(f"  hand-crafted {r_hand:+.3f} -> best learned held-out {best_learned:+.3f}")
    if rd_cv > r2_cv + 0.03:
        log.info(f"  rich features (512-d) add significant value (+{rd_cv-r2_cv:.3f}) -> signal lives in the delta geometry")
    elif rd_cv >= r2_cv - 0.03:
        log.info(f"  rich ~ 2 scalars -> the 2 scalars already capture the main signal, no need for a complex readout")
    else:
        log.info(f"  rich < 2 scalars -> the 512 dims are mostly noise (lost held-out), use the 2-scalar readout")
    log.info("  note: the above is within-perturbation generalization (held-out genes); the cross-perturbation deployment criterion needs a larger cache, tested separately")


if __name__ == "__main__":
    main()
