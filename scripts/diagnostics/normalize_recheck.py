"""
Normalization re-check: raw-count log2fc is suspected to be library-size contaminated (strong cross-KO
anti-correlation; WT predicts at +0.69). Recompute log2fc after per-cell normalization, re-run the
confound decomposition + cross-perturbation correlation, and decide whether +0.69 is purely an artifact.

Features are unchanged (delta/WT/mag from the cache); only the target y changes -> pure NumPy, seconds.
"""
import sys, logging, pickle
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("norm")

from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "FrangiehIzar2021_RNA.h5ad")
CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
N_SPLITS = 5


def cv_r(X, y, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    kf = KFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        s = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(s.transform(X[tr]), y[tr])
        oof[te] = m.predict(s.transform(X[te]))
    return pearsonr(oof, y)[0]


def main():
    # -- normalize + recompute log2fc ---------------------------
    log.info("load + normalize (normalize_total, target_sum=1e4)")
    adata = sc.read_h5ad(DATA)
    keep_g = ["control", "CCND1", "TIMP1", "CDKN2A"]
    sub = adata[(adata.obs["perturbation_2"] == "Control") &
                (adata.obs["perturbation"].isin(keep_g))].copy()
    sc.pp.normalize_total(sub, target_sum=1e4)   # library-size normalization
    var_eids = sub.var["ensembl_id"].values

    def l2fc(gene):
        mc = sub.obs["perturbation"] == "control"
        mk = sub.obs["perturbation"] == gene
        Xc = sub[mc].X; Xk = sub[mk].X
        if sparse.issparse(Xc): Xc, Xk = Xc.toarray(), Xk.toarray()
        return np.log2(Xk.mean(0) + 1) - np.log2(Xc.mean(0) + 1), int(mk.sum())

    vecs = {}
    for g in ["CCND1", "TIMP1", "CDKN2A"]:
        fc, nk = l2fc(g); vecs[g] = fc
        log.info(f"  {g} KO ({nk} cells): log2fc range [{fc.min():.2f}, {fc.max():.2f}]")

    # control split-half null
    mc = sub.obs["perturbation"] == "control"
    ctrl = sub[mc]
    Xc = ctrl.X.toarray() if sparse.issparse(ctrl.X) else np.asarray(ctrl.X)
    rng = np.random.default_rng(0); perm = rng.permutation(ctrl.n_obs); h = ctrl.n_obs // 2
    vecs["control_split"] = np.log2(Xc[perm[:h]].mean(0) + 1) - np.log2(Xc[perm[h:]].mean(0) + 1)
    log.info(f"  control-split null: [{vecs['control_split'].min():.2f}, {vecs['control_split'].max():.2f}]")

    log.info("\n--- cross-perturbation correlation after normalization (expected: low KO-KO and low KO-null) ---")
    nm = list(vecs.keys())
    for i in range(len(nm)):
        for j in range(i + 1, len(nm)):
            log.info(f"  {nm[i]:<14} vs {nm[j]:<14}: r = {pearsonr(vecs[nm[i]], vecs[nm[j]])[0]:+.3f}")

    # -- re-run CCND1 confound decomposition (normalized target) --
    log.info("\n--- CCND1 confound decomposition on the normalized target ---")
    with open(TOKDICT, "rb") as f:
        tokd = pickle.load(f)
    gene2id = {k: v for k, v in tokd.items() if not str(k).startswith("<")}
    gt = {}
    for eid, fc in zip(var_eids, vecs["CCND1"]):
        t = gene2id.get(eid)
        if t is not None:
            gt[int(t)] = float(fc)

    d = np.load(CACHE, allow_pickle=True)
    gene, cellpos, cos, proj = d["gene"], d["cellpos"], d["cos"], d["proj"]
    n = int(d["n_valid"])
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); P = np.full((n, len(uniq)), np.nan, np.float32)
    C[cellpos, col] = cos; P[cellpos, col] = proj
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    proj_by = {int(t): p for t, p in zip(uniq, np.nanmean(P, 0))}
    rich_tokens, mean_delta, mean_wt = d["rich_tokens"], d["mean_delta"], d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}
    common = np.array([int(t) for t in uniq if int(t) in gt and gt[int(t)] != 0 and int(t) in rd])
    y = np.array([gt[int(t)] for t in common])
    di = np.array([rd[int(t)] for t in common])
    D, W = mean_delta[di], mean_wt[di]
    mag = np.array([mag_by[int(t)] for t in common])[:, None]
    pj = np.array([proj_by[int(t)] for t in common])

    log.info(f"  common genes {len(common)}")
    log.info(f"  hand-crafted mag*sign(proj) : {pearsonr(mag[:,0]*np.sign(pj), y)[0]:+.4f}")
    r_wt = cv_r(W, y)
    r_d = cv_r(D, y)
    r_dw = cv_r(np.column_stack([D, W, mag]), y)
    log.info(f"  WT-alone             : held-out {r_wt:+.4f}")
    log.info(f"  delta-alone          : held-out {r_d:+.4f}")
    log.info(f"  delta+WT+mag         : held-out {r_dw:+.4f}")

    log.info("\n===== interpretation =====")
    log.info(f"  delta+WT({r_dw:+.3f}) - WT-alone({r_wt:+.3f}) = {r_dw-r_wt:+.3f} (delta's increment)")
    if r_dw - r_wt > 0.05:
        log.info("  -> delta adds incremental signal after normalization -> perturbation-specific signal exists")
    else:
        log.info("  -> delta still adds no increment -> the embedding perturbation response carries no linearly-readable perturbation-specific signal (for this target)")
    if abs(pearsonr(vecs["CCND1"], vecs["control_split"])[0]) < 0.15:
        log.info("  -> after normalization KO and null are weakly correlated -> the ground truth is clean")
    else:
        log.info("  -> still correlated with the null -> possible residual technical structure")


if __name__ == "__main__":
    main()
