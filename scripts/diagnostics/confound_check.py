"""
Confound check: is the held-out r of ~0.69 real perturbation signal, or leakage from
baseline expression / non-specific structure?

Check 1 (from the CCND1 cache): decompose the readout
  - WT-alone [mean_wt]       -> how much log2fc a pure baseline embedding predicts (high = baseline confound)
  - delta-alone [mean_delta] -> how much the pure perturbation response predicts (this is what we want)
  - delta+WT vs delta-only   -> how much WT adds (the more it adds, the more suspicious)

Check 2 (from adata, no cache needed): is log2fc perturbation-specific?
  - compute KO log2fc for CCND1 / TIMP1 / CDKN2A and their pairwise correlations
  - a control split-half null log2fc (no perturbation): gene-intrinsic noise / expression structure
  - if perturbation log2fcs are highly inter-correlated and resemble the control split -> non-specific,
    so predicting it != predicting the perturbation
"""
import sys, logging
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("confound")

from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "FrangiehIzar2021_RNA.h5ad")
CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")
N_SPLITS = 5


def cv_r(X, y, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    kf = KFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc_ = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc_.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc_.transform(X[te]))
    return pearsonr(oof, y)[0]


def log2fc(adata, ko_gene):
    mc = (adata.obs["perturbation"] == "control") & (adata.obs["perturbation_2"] == "Control")
    mk = (adata.obs["perturbation"] == ko_gene) & (adata.obs["perturbation_2"] == "Control")
    Xc, Xk = adata[mc].X, adata[mk].X
    if sparse.issparse(Xc): Xc, Xk = Xc.toarray(), Xk.toarray()
    return (np.log2(Xk.mean(0) + 1) - np.log2(Xc.mean(0) + 1)), int(mk.sum())


def main():
    # -- Check 1: decompose the readout -------------------------
    log.info("===== Check 1: WT-alone vs delta-alone (CCND1 cache) =====")
    d = np.load(CACHE, allow_pickle=False)
    gene, cellpos, cos, proj = d["gene"], d["cellpos"], d["cos"], d["proj"]
    n = int(d["n_valid"])
    gt = {int(t): float(f) for t, f in zip(d["gt_tokens"], d["gt_fc"])}
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); P = np.full((n, len(uniq)), np.nan, np.float32)
    C[cellpos, col] = cos; P[cellpos, col] = proj
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    rich_tokens, mean_delta, mean_wt = d["rich_tokens"], d["mean_delta"], d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}
    common = np.array([int(t) for t in uniq if int(t) in gt and gt[int(t)] != 0 and int(t) in rd])
    y = np.array([gt[int(t)] for t in common])
    di = np.array([rd[int(t)] for t in common])
    D, W = mean_delta[di], mean_wt[di]
    mag = np.array([mag_by[int(t)] for t in common])[:, None]

    log.info(f"  common genes {len(common)}")
    log.info(f"  WT-alone   [mean_wt(768)]       : held-out {cv_r(W, y):+.4f}  (high = baseline confound)")
    log.info(f"  delta-alone[mean_delta(768)]    : held-out {cv_r(D, y):+.4f}  (pure perturbation response)")
    log.info(f"  delta+mag                       : held-out {cv_r(np.column_stack([D, mag]), y):+.4f}")
    log.info(f"  delta+WT+mag                    : held-out {cv_r(np.column_stack([D, W, mag]), y):+.4f}")

    # -- Check 2: is log2fc perturbation-specific? --------------
    log.info("\n===== Check 2: log2fc perturbation specificity (adata) =====")
    adata = sc.read_h5ad(DATA)
    vecs = {}
    for g in ["CCND1", "TIMP1", "CDKN2A"]:
        try:
            fc, nk = log2fc(adata, g)
            vecs[g] = fc
            log.info(f"  {g} KO: {nk} cells, log2fc range [{fc.min():.2f}, {fc.max():.2f}]")
        except Exception as e:
            log.warning(f"  {g} skipped: {e}")

    # control split-half null
    mc = (adata.obs["perturbation"] == "control") & (adata.obs["perturbation_2"] == "Control")
    ctrl = adata[mc]
    rng = np.random.default_rng(0)
    perm = rng.permutation(ctrl.n_obs)
    h = ctrl.n_obs // 2
    Xc = ctrl.X.toarray() if sparse.issparse(ctrl.X) else np.asarray(ctrl.X)
    null_fc = np.log2(Xc[perm[:h]].mean(0) + 1) - np.log2(Xc[perm[h:]].mean(0) + 1)
    vecs["control_split(null)"] = null_fc
    log.info(f"  control split-half null: log2fc range [{null_fc.min():.2f}, {null_fc.max():.2f}]")

    log.info("\n  --- pairwise Pearson correlation (high = non-specific shared structure) ---")
    names = list(vecs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = pearsonr(vecs[names[i]], vecs[names[j]])[0]
            log.info(f"  {names[i]:<22} vs {names[j]:<22}: r = {r:+.3f}")

    log.info("\n===== interpretation =====")
    log.info("  WT-alone high -> most of the +0.69 is baseline-expression confound, not perturbation signal")
    log.info("  KO log2fcs highly inter-correlated / close to the control split -> log2fc is not perturbation-specific")
    log.info("  delta-alone still clearly positive and KO log2fcs weakly inter-correlated -> real perturbation signal")


if __name__ == "__main__":
    main()
