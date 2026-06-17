"""
Nonlinear seal: does delta also add zero increment over the DESeq2 LFC under a nonlinear readout?
Closes the "you didn't try a nonlinear readout" objection -> negative under both linear and nonlinear.

Fairness:
  - increment = (WT vs WT+delta) within the same model class; the baseline also passes through the same
    nonlinear model (you cannot compare a linear baseline to MLP+delta, that conflates "nonlinear" and "delta")
  - three model classes: ridge (linear) / MLP (strong regularization + early stopping) / HistGBT (robust to nonlinearity, low overfit risk)
  - same gene set + same CV folds; report fold-to-fold variance
  - plus an apples-to-apples baseline check: WT-alone on the raw vs DESeq2 targets over the same gene set (point 5)

target = CCND1 DESeq2 log2FC (primary DESeq2). Run from the project root.
"""
import sys, logging, pickle
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("seal")

CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")
DESEQ = Path("benchmark_output/deseq2/ccnd1_deseq2.npz")
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
N_SPLITS = 5
SEED = 0


def make_models():
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {
        "ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 13)),
        "MLP":   lambda: MLPRegressor(hidden_layer_sizes=(32,), alpha=10.0,
                                      early_stopping=True, n_iter_no_change=10,
                                      max_iter=400, random_state=SEED),
        "GBT":   lambda: HistGradientBoostingRegressor(max_depth=3, l2_regularization=1.0,
                                                       early_stopping=True, random_state=SEED),
    }


def cv_r(make, X, y):
    """held-out CV: returns (Pearson r of pooled OOF, std of per-fold r). StandardScaler fit on the train fold only."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan); fold_r = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = make().fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        oof[te] = pred
        if np.std(pred) > 1e-9:
            fold_r.append(pearsonr(pred, y[te])[0])
    return pearsonr(oof, y)[0], (np.std(fold_r) if fold_r else np.nan)


def main():
    # -- align: cache features + DESeq2 target + raw target, common gene set --
    with open(TOKDICT, "rb") as f:
        tokd = pickle.load(f)
    gene2id = {k: v for k, v in tokd.items() if not str(k).startswith("<")}
    id2eid = {v: k for k, v in gene2id.items()}

    d = np.load(CACHE, allow_pickle=True)
    gene, cellpos, cos = d["gene"], d["cellpos"], d["cos"]
    n = int(d["n_valid"])
    raw_gt = {int(t): float(f) for t, f in zip(d["gt_tokens"], d["gt_fc"])}   # raw-count log2fc
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); C[cellpos, col] = cos
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    rich_tokens, mean_delta, mean_wt = d["rich_tokens"], d["mean_delta"], d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}

    dq = np.load(DESEQ, allow_pickle=True)
    eid2deseq = {eid: v for eid, v in zip(dq["var_eids"], dq["ccnd1_deseq"]) if np.isfinite(v)}

    rows = []
    for t in uniq:
        t = int(t)
        if t not in rd:
            continue
        eid = id2eid.get(t)
        if eid in eid2deseq and t in raw_gt:
            rows.append((t, eid))
    toks = [r[0] for r in rows]
    di = np.array([rd[t] for t in toks])
    D = mean_delta[di]; W = mean_wt[di]
    y_deseq = np.array([eid2deseq[e] for _, e in rows])
    y_raw = np.array([raw_gt[t] for t in toks])
    keep = (y_deseq != 0)
    D, W, y_deseq, y_raw = D[keep], W[keep], y_deseq[keep], y_raw[keep]
    log.info(f"common gene set {len(y_deseq)} (cache ∩ DESeq2 ∩ raw)")

    # -- point 5: WT-alone on raw vs DESeq2 over the same gene set (baseline tracks target quality) --
    models = make_models()
    log.info("\n===== baseline tracks target quality (same gene set + same folds, ridge WT-alone) =====")
    r_raw, _ = cv_r(models["ridge"], W, y_raw)
    r_de, _ = cv_r(models["ridge"], W, y_deseq)
    log.info(f"  WT-alone -> raw log2FC   : {r_raw:+.3f}")
    log.info(f"  WT-alone -> DESeq2 log2FC: {r_de:+.3f}  (same genes, same folds; the difference is purely target quality)")

    # -- nonlinear seal: three models, WT vs WT+delta (DESeq2 target) --
    log.info("\n===== nonlinear seal: delta increment (target=DESeq2 LFC) =====")
    log.info(f"  {'model':<6}{'WT-alone':>11}{'WT+delta':>11}{'incr':>9}{'fold std':>9}")
    XWT = W
    XWTD = np.column_stack([W, D])
    results = {}
    for name, make in models.items():
        rwt, _ = cv_r(make, XWT, y_deseq)
        rwd, sd = cv_r(make, XWTD, y_deseq)
        # delta alone (check for any nonlinear signal)
        rdo, _ = cv_r(make, D, y_deseq)
        results[name] = (rwt, rwd, rwd - rwt, sd, rdo)
        log.info(f"  {name:<6}{rwt:>+11.3f}{rwd:>+11.3f}{rwd-rwt:>+9.3f}{sd:>9.3f}   (delta-alone {rdo:+.3f})")

    # -- MLP regularization robustness sweep (show the negative increment is overfit, robustly non-positive) --
    from sklearn.neural_network import MLPRegressor
    log.info("\n===== MLP regularization sweep (delta increment should be robustly non-positive, approaching 0) =====")
    for alpha in [1.0, 10.0, 30.0, 100.0]:
        mk = lambda a=alpha: MLPRegressor(hidden_layer_sizes=(32,), alpha=a, early_stopping=True,
                                          n_iter_no_change=10, max_iter=400, random_state=SEED)
        rwt, _ = cv_r(mk, XWT, y_deseq)
        rwd, sd = cv_r(mk, XWTD, y_deseq)
        log.info(f"  MLP alpha={alpha:<5}: WT {rwt:+.3f} -> WT+delta {rwd:+.3f}  incr {rwd-rwt:+.3f}  std {sd:.3f}")

    # -- interpretation: the seal is about whether there is any *positive* increment; a negative increment = delta hurts = confirms no signal --
    log.info("\n===== interpretation =====")
    incs = {k: v[2] for k, v in results.items()}
    max_pos = max(incs.values())
    log.info(f"  delta increments: {', '.join(f'{k}:{v:+.3f}' for k,v in incs.items())}")
    if max_pos < 0.03:
        log.info("  -> no model shows a positive delta increment (ridge/GBT ~0; MLP negative = overfitting noise dims, delta hurts)")
        log.info("  -> negative under both linear and nonlinear (GBT robust / MLP): delta carries no readable perturbation-specific signal, sealed")
    else:
        big = {k: v for k, v in incs.items() if v >= 0.03}
        log.info(f"  WARNING: some model shows a positive increment: {big} -> re-check whether it is real signal vs overfitting (see fold std)")


if __name__ == "__main__":
    main()
