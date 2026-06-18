"""
Path 2 (target switched to "which genes are affected" / saliency) + perturbation-specificity control.

Core caution: hub / highly expressed genes shift the embedding for any perturbation and are often
called DEGs. An "affected-gene ranking" can easily predict the top DEGs while only capturing the
generic "responds to anything" set, not perturbation-specific signal. So a "universal responsiveness"
baseline must be controlled for; only predicting CCND1's *specific* effect counts as real signal.

Design:
  universal responsiveness universal_abs[g] = mean of normalized |log2fc| across all Frangieh perturbations (baseline confound)
  CCND1 effect ccnd1_abs[g]                 = |log2fc_CCND1(g)|
  features: delta_mag (1-cos), ||mean_delta||, rich mean_delta(768); confounds: WT(768), universal_abs

  Key tests:
   (1) delta_mag vs ccnd1_abs            (naive saliency; expected positive but confounded)
   (2) delta_mag vs universal_abs        (strength of the universal confound)
   (3) partial(delta_mag, ccnd1_abs | universal_abs)  (specific saliency after removing universal = key scalar)
   (4) ridge held-out: baseline[universal,WT] vs +mean_delta -> delta increment (key learned readout)

Targets use normalized data (normalize_total); primary DESeq2 needs pseudobulk+DESeq2 (flagged).
"""
import sys, logging, pickle
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, spearmanr, rankdata
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("path2")

from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "FrangiehIzar2021_RNA.h5ad")
CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
MIN_CELLS = 30
N_SPLITS = 5


def partial_spearman(a, b, c):
    """partial Spearman(a,b | c): linear residual correlation after rank transform."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    def resid(y, x):
        x1 = np.column_stack([x, np.ones_like(x)])
        beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
        return y - x1 @ beta
    return pearsonr(resid(ra, rc), resid(rb, rc))[0]


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
    # -- 1. normalize + all-perturbation log2fc -> universal responsiveness --
    log.info("Step 1: normalize + compute all-perturbation log2fc (universal-responsiveness baseline)")
    adata = sc.read_h5ad(DATA)
    sub = adata[adata.obs["perturbation_2"] == "Control"].copy()
    sc.pp.normalize_total(sub, target_sum=1e4)
    var_eids = sub.var["ensembl_id"].values

    perts = sub.obs["perturbation"].value_counts()
    perts = [p for p in perts.index if p != "control" and perts[p] >= MIN_CELLS]
    log.info(f"  {len(perts)} perturbations (>={MIN_CELLS} cells)")

    mc = (sub.obs["perturbation"] == "control").values
    Xc = sub.X[mc]
    Xc = Xc.toarray() if sparse.issparse(Xc) else np.asarray(Xc)
    mean_ctrl = Xc.mean(0)
    logctrl = np.log2(mean_ctrl + 1)

    abs_mat = np.zeros((len(perts), sub.n_vars), dtype=np.float32)
    ccnd1_signed = None
    for i, p in enumerate(perts):
        mk = (sub.obs["perturbation"] == p).values
        Xk = sub.X[mk]; Xk = Xk.toarray() if sparse.issparse(Xk) else np.asarray(Xk)
        fc = np.log2(Xk.mean(0) + 1) - logctrl
        abs_mat[i] = np.abs(fc)
        if p == "CCND1":
            ccnd1_signed = fc
    universal_abs = abs_mat.mean(0)              # universal responsiveness
    ccnd1_abs = np.abs(ccnd1_signed)
    log.info(f"  universal responsiveness vs CCND1|log2fc| correlation: {spearmanr(universal_abs, ccnd1_abs)[0]:+.3f}")

    # -- 2. load CCND1 cache features ---------------------------
    log.info("Step 2: load CCND1 delta features")
    with open(TOKDICT, "rb") as f:
        tokd = pickle.load(f)
    gene2id = {k: v for k, v in tokd.items() if not str(k).startswith("<")}
    eid2col = {eid: i for i, eid in enumerate(var_eids)}

    d = np.load(CACHE, allow_pickle=False)
    gene, cellpos, cos = d["gene"], d["cellpos"], d["cos"]
    n = int(d["n_valid"])
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); C[cellpos, col] = cos
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    rich_tokens, mean_delta, mean_wt = d["rich_tokens"], d["mean_delta"], d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}
    id2eid = {v: k for k, v in gene2id.items()}

    # align: tokens with delta and mag that also map to an adata gene
    rows = []
    for t in uniq:
        t = int(t)
        if t not in rd or t not in mag_by:
            continue
        eid = id2eid.get(t)
        if eid not in eid2col:
            continue
        rows.append((t, eid2col[eid]))
    toks = np.array([r[0] for r in rows]); gcols = np.array([r[1] for r in rows])
    di = np.array([rd[t] for t in toks])
    D = mean_delta[di]; W = mean_wt[di]
    dmag = np.array([mag_by[t] for t in toks])           # 1-cos magnitude
    dnorm = np.linalg.norm(D, axis=1)                    # ||mean_delta||
    uabs = universal_abs[gcols]; cabs = ccnd1_abs[gcols]
    log.info(f"  aligned genes {len(toks)}")

    # -- 3. key tests -------------------------------------------
    log.info("\n===== key tests (target = CCND1 |log2fc| affectedness) =====")
    log.info(f"(1) delta_mag(1-cos) vs CCND1|fc|        : Spearman {spearmanr(dmag, cabs)[0]:+.3f}  (naive)")
    log.info(f"  ||delta||          vs CCND1|fc|        : Spearman {spearmanr(dnorm, cabs)[0]:+.3f}")
    log.info(f"(2) delta_mag        vs universal resp.  : Spearman {spearmanr(dmag, uabs)[0]:+.3f}  (universal confound)")
    log.info(f"  ||delta||          vs universal resp.  : Spearman {spearmanr(dnorm, uabs)[0]:+.3f}")
    log.info(f"(3) partial(delta_mag, CCND1|fc| | univ) : {partial_spearman(dmag, cabs, uabs):+.3f}  *specific saliency after removing universal")
    log.info(f"  partial(||delta||, CCND1|fc| | univ)   : {partial_spearman(dnorm, cabs, uabs):+.3f}")

    # -- 4. learned: delta increment over the specific effect ---
    log.info("\n===== learned held-out (target=CCND1|fc|, controlling for universal+WT baseline) =====")
    y = cabs
    base = np.column_stack([uabs, W])                    # universal responsiveness + WT baseline (confounds)
    r_base = cv_r(base, y)
    r_full = cv_r(np.column_stack([uabs, W, D]), y)
    log.info(f"  baseline [universal, WT]     : held-out {r_base:+.4f}")
    log.info(f"  + mean_delta(768)            : held-out {r_full:+.4f}")
    log.info(f"  delta increment              : {r_full - r_base:+.4f}")

    log.info("\n===== interpretation =====")
    p3 = partial_spearman(dmag, cabs, uabs)
    if p3 > 0.1 and (r_full - r_base) > 0.03:
        log.info(f"  -> after removing universal responsiveness, delta still predicts CCND1's specific effect -> Path 2 has real perturbation-specific saliency")
    elif abs(p3) < 0.05 and abs(r_full - r_base) < 0.02:
        log.info(f"  -> delta only tracks universal responsiveness; nothing CCND1-specific after removal -> Path 2 is also confounded, not real signal")
    else:
        log.info(f"  -> weak/borderline signal, needs more perturbations to confirm")


if __name__ == "__main__":
    main()
