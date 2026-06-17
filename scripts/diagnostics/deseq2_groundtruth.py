"""
Primary ground truth: recompute Frangieh log2FC with pseudobulk + DESeq2 (size factors),
and re-run the three-readout increment diagnostic (WT-alone / delta-alone / delta+WT) on the clean target.

Methodology (defensible to reviewers):
  - pseudobulk: aggregate raw counts by (perturbation x sgRNA); each sgRNA = one biological replicate
    (median 4 replicates per perturbation, minimum 2; Squair 2021 recommends pseudobulk + DESeq2/edgeR)
  - CCND1 headline target: full pydeseq2 DESeq2 (median-of-ratios size factors + dispersion + apeglm LFC shrinkage)
  - universal-responsiveness baseline: mean |LFC| of per-perturbation log2FC from size-factor-normalized pseudobulk

Expectation: the delta increment is still ~0 (the WT-alone control is a relative comparison, robust to
target contamination), now at DESeq2 grade.

Run from the project root: python scripts/diagnostics/deseq2_groundtruth.py
"""
import sys, logging, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("deseq2")

from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "FrangiehIzar2021_RNA.h5ad")
CACHE = Path("benchmark_output/pathd_cache/CCND1_cache.npz")
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
OUTDIR = Path("benchmark_output/deseq2"); OUTDIR.mkdir(parents=True, exist_ok=True)
MIN_CELLS = 10      # minimum cells per pseudobulk sample
MIN_TOTAL = 10      # minimum total count per gene
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


def size_factors(counts):
    """DESeq2 median-of-ratios size factors. counts: samples x genes."""
    logc = np.log(counts.astype(np.float64))
    # geometric mean using only genes that are > 0 in all samples
    gene_ok = np.all(counts > 0, axis=0)
    log_gmean = logc[:, gene_ok].mean(axis=0)          # per-gene geometric mean (log)
    ratios = logc[:, gene_ok] - log_gmean[None, :]     # log ratio
    sf = np.exp(np.median(ratios, axis=1))             # per-sample median
    return sf / np.exp(np.mean(np.log(sf)))            # center


def build_pseudobulk(adata):
    """Aggregate raw counts by (perturbation x sgRNA) -> samples x genes + sample metadata."""
    o = adata.obs
    X = adata.X
    if sparse.issparse(X): X = X.tocsr()
    key = o["perturbation"].astype(str) + "||" + o["sgRNA"].astype(str)
    groups = pd.Categorical(key)
    samples, pert_of, celln = [], [], []
    rows = []
    for code, name in enumerate(groups.categories):
        idx = np.where(groups.codes == code)[0]
        if len(idx) < MIN_CELLS:
            continue
        pert = name.split("||")[0]
        sub = X[idx]
        s = np.asarray(sub.sum(axis=0)).ravel() if sparse.issparse(sub) else sub.sum(0)
        rows.append(s); samples.append(name); pert_of.append(pert); celln.append(len(idx))
    counts = np.vstack(rows).astype(np.int64)          # samples x genes
    meta = pd.DataFrame({"sample": samples, "perturbation": pert_of, "n_cells": celln})
    return counts, meta


def main():
    log.info("Step 1: load + build pseudobulk (Control condition, by sgRNA)")
    adata = sc.read_h5ad(DATA)
    adata = adata[adata.obs["perturbation_2"] == "Control"].copy()
    counts, meta = build_pseudobulk(adata)
    var_eids = adata.var["ensembl_id"].values
    log.info(f"  pseudobulk samples {counts.shape[0]}, genes {counts.shape[1]}; perturbations {meta['perturbation'].nunique()}")

    # gene filter
    gene_keep = counts.sum(0) >= MIN_TOTAL
    counts = counts[:, gene_keep]; var_eids = var_eids[gene_keep]
    log.info(f"  after gene filter {counts.shape[1]} (total count >= {MIN_TOTAL})")

    # -- Step 2: size-factor normalization -> universal responsiveness + per-pert LFC --
    log.info("Step 2: median-of-ratios size factors + per-perturbation log2FC")
    sf = size_factors(counts)
    norm = counts / sf[:, None]                         # normalized pseudobulk
    perts = [p for p in meta["perturbation"].unique() if p != "control"]
    ctrl_mean = norm[(meta["perturbation"] == "control").values].mean(0)
    logctrl = np.log2(ctrl_mean + 1)
    lfc = {}
    for p in perts:
        m = (meta["perturbation"] == p).values
        if m.sum() < 2:
            continue
        lfc[p] = np.log2(norm[m].mean(0) + 1) - logctrl
    log.info(f"  {len(lfc)} perturbations have a size-factor LFC with >=2 replicates")
    universal_abs = np.mean([np.abs(v) for v in lfc.values()], axis=0)
    ccnd1_sf = lfc["CCND1"]

    # -- Step 3: full CCND1 DESeq2 (headline) -------------------
    log.info("Step 3: full CCND1 DESeq2 (size factors + dispersion + apeglm shrinkage)")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    sel = meta["perturbation"].isin(["CCND1", "control"]).values
    cdf = pd.DataFrame(counts[sel], index=np.array(meta["sample"])[sel],
                       columns=[f"g{i}" for i in range(counts.shape[1])])
    # control set as the reference level (first category) -> coefficient condition[T.CCND1], correct sign
    cond = pd.Categorical(np.where(meta["perturbation"][sel].values == "CCND1", "CCND1", "control"),
                          categories=["control", "CCND1"])
    mdf = pd.DataFrame({"condition": cond}, index=cdf.index)
    dds = DeseqDataSet(counts=cdf, metadata=mdf, design="~condition", quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", "CCND1", "control"], quiet=True)
    st.summary()   # explicit contrast -> results_df has the CCND1-vs-control LFC
    log.info(f"  LFC coefficient columns: {list(dds.varm['LFC'].columns)}")
    cand = [c for c in dds.varm["LFC"].columns if "CCND1" in c]   # should be condition[T.CCND1]
    if cand:
        try:
            st.lfc_shrink(coeff=cand[0]); log.info(f"  LFC shrunk ({cand[0]})")
        except Exception as e:
            log.warning(f"  LFC shrinkage skipped ({e}); using unshrunk Wald LFC")
    else:
        log.warning("  CCND1 coefficient not found; using unshrunk Wald LFC (sign guaranteed by the explicit contrast)")
    ccnd1_deseq = st.results_df["log2FoldChange"].values
    # sign sanity: DESeq2 LFC should share a sign with the size-factor LFC (CCND1-vs-control)
    _ok = np.isfinite(ccnd1_deseq)
    sign_r = pearsonr(ccnd1_deseq[_ok], ccnd1_sf[_ok])[0]
    log.info(f"  sign sanity: corr(DESeq2 LFC, size-factor LFC) = {sign_r:+.3f} (should be strongly positive)")
    if sign_r < 0:
        log.warning("  WARNING: opposite sign! The DESeq2 LFC direction may be flipped; check the reference level")
    log.info(f"  CCND1 DESeq2 log2FC range [{np.nanmin(ccnd1_deseq):.2f}, {np.nanmax(ccnd1_deseq):.2f}]")

    # -- Step 4: load cache features + align --------------------
    log.info("Step 4: increment diagnostic (WT-alone / delta-alone / delta+WT), DESeq2 target")
    with open(TOKDICT, "rb") as f:
        tokd = pickle.load(f)
    gene2id = {k: v for k, v in tokd.items() if not str(k).startswith("<")}
    eid2col = {eid: i for i, eid in enumerate(var_eids)}

    d = np.load(CACHE, allow_pickle=True)
    gene, cellpos, cos = d["gene"], d["cellpos"], d["cos"]
    n = int(d["n_valid"])
    uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
    col = np.array([t2c[t] for t in gene])
    C = np.full((n, len(uniq)), np.nan, np.float32); C[cellpos, col] = cos
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    rich_tokens, mean_delta, mean_wt = d["rich_tokens"], d["mean_delta"], d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}
    id2eid = {v: k for k, v in gene2id.items()}

    # targets: DESeq2 LFC / size-factor LFC, aligned by token
    rows = []
    for t in uniq:
        t = int(t)
        if t not in rd or t not in mag_by: continue
        eid = id2eid.get(t)
        if eid not in eid2col: continue
        rows.append((t, eid2col[eid]))
    toks = np.array([r[0] for r in rows]); gcols = np.array([r[1] for r in rows])
    di = np.array([rd[t] for t in toks])
    D, W = mean_delta[di], mean_wt[di]
    mag = np.array([mag_by[t] for t in toks])[:, None]
    y_deseq = ccnd1_deseq[gcols]
    uabs = universal_abs[gcols]
    keep = ~np.isnan(y_deseq) & (y_deseq != 0)
    D, W, mag, uabs = D[keep], W[keep], mag[keep], uabs[keep]
    y_deseq = y_deseq[keep]
    log.info(f"  aligned genes {keep.sum()}")

    # -- Step 5: increment diagnostic (DESeq2 log2FC target) ----
    r_wt = cv_r(W, y_deseq)
    r_d = cv_r(D, y_deseq)
    r_dw = cv_r(np.column_stack([D, W, mag]), y_deseq)
    log.info("\n===== increment diagnostic on the DESeq2 log2FC target =====")
    log.info(f"  WT-alone        : held-out {r_wt:+.4f}")
    log.info(f"  delta-alone     : held-out {r_d:+.4f}")
    log.info(f"  delta+WT+mag    : held-out {r_dw:+.4f}")
    log.info(f"  delta increment : {r_dw - r_wt:+.4f}")

    # saliency view
    yabs = np.abs(y_deseq)
    base = np.column_stack([uabs, W])
    r_sal_base = cv_r(base, yabs)
    r_sal_full = cv_r(np.column_stack([uabs, W, D]), yabs)
    log.info(f"\n  [saliency] baseline[universal,WT] {r_sal_base:+.4f} -> +delta {r_sal_full:+.4f} (increment {r_sal_full-r_sal_base:+.4f})")

    # -- compare + save -----------------------------------------
    log.info("\n===== three-target increment comparison (old vs DESeq2) =====")
    log.info(f"  {'target':<22}{'WT-alone':>10}{'+delta':>10}{'incr':>9}")
    log.info(f"  {'raw log2FC (contam.)':<22}{'+0.693':>10}{'+0.690':>10}{'-0.003':>9}")
    log.info(f"  {'normalize_total':<22}{'+0.381':>10}{'+0.376':>10}{'-0.005':>9}")
    log.info(f"  {'DESeq2 (this run)':<22}{r_wt:>+10.3f}{r_dw:>+10.3f}{r_dw-r_wt:>+9.3f}")

    np.savez_compressed(OUTDIR / "ccnd1_deseq2.npz",
                        ccnd1_deseq=ccnd1_deseq, universal_abs=universal_abs,
                        var_eids=var_eids, ccnd1_sizefactor=ccnd1_sf)
    log.info(f"\nresult saved: {OUTDIR/'ccnd1_deseq2.npz'}")
    if abs(r_dw - r_wt) < 0.03:
        log.info("-> delta increment still ~0 on a DESeq2-grade target: the negative conclusion survives primary recomputation")
    else:
        log.info("-> WARNING: the increment changed materially on the DESeq2 target, re-check")


if __name__ == "__main__":
    main()
