"""
Thin-gate [mag,proj] truth check: held-out CV r on both the *contaminated raw* and the *clean DESeq2*
targets. Answers the key Fig 1B question --
"the +0.27 thin gate: does it actually collapse to zero under the correct control
(library-size normalization -> clean target)?"

Features: mag = mean_cell(1-cos), proj = mean_cell(delta . expr_axis) (same as thin_gate.py).
Alignment: cache token -> ensembl id (token_dictionary) -> DESeq2 LFC (ccnd1_deseq2.npz).
Common gene set, same folds, ridge held-out (same protocol as confound_check / probe_pathd).
Run from the project root.
"""
import sys, logging, pickle
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("thin_clean")

CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")
DESEQ = Path("benchmark_output/deseq2/ccnd1_deseq2.npz")
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
N_SPLITS = 5
SEED = 0


def cv_r(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    alphas = np.logspace(-2, 4, 13)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=alphas).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return pearsonr(oof, y)[0]


def main():
    with open(TOKDICT, "rb") as f:
        tokd = pickle.load(f)
    gene2id = {k: v for k, v in tokd.items() if not str(k).startswith("<")}
    id2eid = {v: k for k, v in gene2id.items()}

    d = np.load(CACHE, allow_pickle=False)
    gene, cellpos, cos, proj = d["gene"], d["cellpos"], d["cos"], d["proj"]
    n = int(d["n_valid"])
    raw_gt = {int(t): float(f) for t, f in zip(d["gt_tokens"], d["gt_fc"])}
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); P = np.full((n, len(uniq)), np.nan, np.float32)
    C[cellpos, col] = cos; P[cellpos, col] = proj
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    proj_by = {int(t): p for t, p in zip(uniq, np.nanmean(P, 0))}

    dq = np.load(DESEQ, allow_pickle=False)
    eid2deseq = {eid: v for eid, v in zip(dq["var_eids"], dq["ccnd1_deseq"]) if np.isfinite(v)}

    # common gene set: cache ∩ DESeq2 ∩ raw (same protocol as nonlinear_seal)
    toks = []
    for t in uniq:
        t = int(t)
        eid = id2eid.get(t)
        if eid in eid2deseq and t in raw_gt:
            toks.append(t)
    mag = np.array([mag_by[t] for t in toks])
    pj = np.array([proj_by[t] for t in toks])
    y_raw = np.array([raw_gt[t] for t in toks])
    y_deseq = np.array([eid2deseq[id2eid[t]] for t in toks])
    keep = (y_deseq != 0) & (y_raw != 0)
    mag, pj, y_raw, y_deseq = mag[keep], pj[keep], y_raw[keep], y_deseq[keep]
    log.info(f"common gene set {len(y_deseq)} (cache ∩ DESeq2 ∩ raw, gt!=0)")

    X2 = np.column_stack([mag, pj])
    log.info("\n===== thin gate [mag,proj] held-out CV r (same gene set, same folds) =====")
    log.info(f"  {'target':<26}{'mag*sign(proj)':>16}{'proj only':>11}{'[mag,proj]':>12}")
    for name, y in [("raw log2FC (contaminated)", y_raw), ("DESeq2 log2FC (clean)", y_deseq)]:
        r_hand = pearsonr(mag * np.sign(pj), y)[0]
        r_proj = pearsonr(pj, y)[0]
        r2 = cv_r(X2, y)
        log.info(f"  {name:<26}{r_hand:>+16.4f}{r_proj:>+11.4f}{r2:>+12.4f}")

    log.info("\n  note: the raw column should reproduce [mag,proj] held-out ~ +0.24 (probe_pathd).")
    log.info("  the DESeq2 column = the thin-gate truth value after the correct control (library-size normalization, sec 2.3).")


if __name__ == "__main__":
    main()
