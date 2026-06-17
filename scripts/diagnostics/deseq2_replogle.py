"""
Replogle primary ground truth + increment diagnostic (parameterized by gene).

Differences from the Frangieh version (deseq2_groundtruth.py):
  - pseudobulk replicates = (perturbation x batch) (Replogle's 48 gemgroup batches), not sgRNA
  - control = non-targeting (perturbation == 'control', confirmed NTC); no perturbation_2 condition
  - parameterized by GENE; the universal-responsiveness baseline is cached once
    (replogle_universal.npz) and reused across the 5 genes

GATA1 caveat: few cells (108, ~2/batch) -> few batch replicates, weaker evidence; not load-bearing,
treated only as a "strongest-signal encore".

Run from the project root: python scripts/diagnostics/deseq2_replogle.py HSPA9
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
log = logging.getLogger("deseq2-rep")

GENE = sys.argv[1] if len(sys.argv) > 1 else "HSPA9"
from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "ReplogleWeissman2022_K562_essential.h5ad")
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
OUTDIR = Path("benchmark_output/deseq2"); OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = Path("benchmark_output/pathd_cache")
UNIV = OUTDIR / "replogle_universal.npz"
MIN_SAMPLE = 5      # minimum cells per batch-pseudobulk sample
MIN_TOTAL = 10
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
    logc = np.log(counts.astype(np.float64) + 0.5)
    gene_ok = np.all(counts > 0, axis=0)
    if gene_ok.sum() < 50:
        gene_ok = counts.min(0) > 0
    log_gmean = logc[:, gene_ok].mean(axis=0)
    ratios = logc[:, gene_ok] - log_gmean[None, :]
    sf = np.exp(np.median(ratios, axis=1))
    return sf / np.exp(np.mean(np.log(sf)))


def build_universal(adata, var_eids):
    """per-perturbation pseudobulk (sum all cells per perturbation) -> size-factor LFC -> universal responsiveness."""
    log.info("building the universal-responsiveness baseline (per-perturbation pseudobulk)...")
    pert = adata.obs["perturbation"].astype(str).values
    cats = [p for p in pd.unique(pert) if (pert == p).sum() >= 50]
    X = adata.X.tocsr() if sparse.issparse(adata.X) else adata.X
    rows, names = [], []
    for p in cats:
        idx = np.where(pert == p)[0]
        s = np.asarray(X[idx].sum(0)).ravel() if sparse.issparse(X) else X[idx].sum(0)
        rows.append(s); names.append(p)
    counts = np.vstack(rows).astype(np.int64)
    keep = counts.sum(0) >= MIN_TOTAL
    counts = counts[:, keep]; eids = var_eids[keep]
    sf = size_factors(counts)
    norm = counts / sf[:, None]
    ci = names.index("control")
    logctrl = np.log2(norm[ci] + 1)
    lfc = np.log2(norm + 1) - logctrl[None, :]      # [perts, genes]
    nonctrl = [i for i, n in enumerate(names) if n != "control"]
    universal_abs = np.abs(lfc[nonctrl]).mean(0)
    np.savez_compressed(UNIV, eids=eids, universal_abs=universal_abs,
                        names=np.array(names), lfc=lfc.astype(np.float32))
    log.info(f"  universal-responsiveness cache: {UNIV} ({len(nonctrl)} perturbations)")
    return eids, universal_abs, {n: lfc[i] for i, n in enumerate(names)}


def main():
    log.info(f"GENE={GENE}")
    adata = sc.read_h5ad(DATA)
    var_eids = adata.var["ensembl_id"].values

    # -- universal responsiveness (reuse cache) --
    if UNIV.exists():
        u = np.load(UNIV, allow_pickle=True)
        eids_u, universal_abs = u["eids"], u["universal_abs"]
        sf_lfc = {n: u["lfc"][i] for i, n in enumerate(u["names"])}
        log.info(f"  loaded universal-responsiveness cache ({UNIV})")
    else:
        eids_u, universal_abs, sf_lfc = build_universal(adata, var_eids)
    eid2ucol = {e: i for i, e in enumerate(eids_u)}

    # -- GENE DESeq2 (batch pseudobulk, 2 groups) --
    log.info("GENE DESeq2 ((perturbation x batch) pseudobulk, control reference)")
    o = adata.obs
    sel_cells = o["perturbation"].isin([GENE, "control"]).values
    sub = adata[sel_cells]
    pert = sub.obs["perturbation"].astype(str).values
    batch = sub.obs["batch"].astype(str).values
    X = sub.X.tocsr() if sparse.issparse(sub.X) else sub.X
    key = pd.Series([f"{p}||{b}" for p, b in zip(pert, batch)])
    rows, smeta = [], []
    for k, idx in key.groupby(key).groups.items():
        ii = np.asarray(idx)
        if len(ii) < MIN_SAMPLE:
            continue
        s = np.asarray(X[ii].sum(0)).ravel() if sparse.issparse(X) else X[ii].sum(0)
        rows.append(s); smeta.append(k.split("||")[0])
    counts = np.vstack(rows).astype(np.int64)
    cond = np.array(smeta)
    n_gene_rep = int((cond == GENE).sum()); n_ctrl_rep = int((cond == "control").sum())
    log.info(f"  pseudobulk replicates: {GENE}={n_gene_rep}, control={n_ctrl_rep}")
    if n_gene_rep < 2:
        log.error(f"  {GENE} has <2 batch replicates, cannot run DESeq2"); return
    if n_gene_rep < 3:
        log.warning(f"  WARNING: {GENE} has only {n_gene_rep} replicates (few cells), weaker evidence -- encore only, not load-bearing")

    keep = counts.sum(0) >= MIN_TOTAL
    counts = counts[:, keep]; eids = var_eids[keep]

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    cdf = pd.DataFrame(counts, index=[f"s{i}" for i in range(len(counts))],
                       columns=[f"g{i}" for i in range(counts.shape[1])])
    cat = pd.Categorical(cond, categories=["control", GENE])   # control = reference
    mdf = pd.DataFrame({"condition": cat}, index=cdf.index)
    dds = DeseqDataSet(counts=cdf, metadata=mdf, design="~condition", quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", GENE, "control"], quiet=True)
    st.summary()
    cand = [c for c in dds.varm["LFC"].columns if GENE in c]
    if cand:
        try: st.lfc_shrink(coeff=cand[0]); log.info(f"  LFC shrunk ({cand[0]})")
        except Exception as e: log.warning(f"  LFC shrinkage skipped ({e})")
    gene_deseq = st.results_df["log2FoldChange"].values

    # sign-check (explicit): DESeq2 LFC vs size-factor LFC, same sign
    sf_g = sf_lfc.get(GENE)
    if sf_g is not None:
        sfg_by_eid = {e: sf_g[eid2ucol[e]] for e in eids if e in eid2ucol}
        a = np.array([gene_deseq[i] for i, e in enumerate(eids) if e in sfg_by_eid])
        b = np.array([sfg_by_eid[e] for e in eids if e in sfg_by_eid])
        ok = np.isfinite(a) & np.isfinite(b)
        sr = pearsonr(a[ok], b[ok])
        log.info(f"  sign-check: corr(DESeq2, size-factor LFC) Pearson {sr[0]:+.3f} (should be strongly positive)")
        if sr[0] < 0: log.warning("  WARNING: opposite sign! check the reference level")

    # -- increment diagnostic --
    log.info("increment diagnostic (DESeq2 target)")
    with open(TOKDICT, "rb") as f:
        gene2id = {k: v for k, v in pickle.load(f).items() if not str(k).startswith("<")}
    id2eid = {v: k for k, v in gene2id.items()}
    eid2col = {e: i for i, e in enumerate(eids)}

    cache = CACHEDIR / f"{GENE}_cache.npz"
    if not cache.exists():
        log.error(f"  cache not found: {cache} (path-D cache incomplete)"); return
    d = np.load(cache, allow_pickle=True)
    g, cp, cs = d["gene"], d["cellpos"], d["cos"]; n = int(d["n_valid"])
    uniq = np.unique(g); t2c = {t: i for i, t in enumerate(uniq)}
    C = np.full((n, len(uniq)), np.nan, np.float32); C[cp, np.array([t2c[t] for t in g])] = cs
    mag_by = {int(t): m for t, m in zip(uniq, np.nanmean(1 - C, 0))}
    rich_tokens, mean_delta, mean_wt = d["rich_tokens"], d["mean_delta"], d["mean_wt"]
    rd = {int(t): i for i, t in enumerate(rich_tokens)}

    rows2 = []
    for t in uniq:
        t = int(t)
        if t not in rd or t not in mag_by: continue
        e = id2eid.get(t)
        if e in eid2col and e in eid2ucol: rows2.append((t, e))
    toks = [r[0] for r in rows2]
    di = np.array([rd[t] for t in toks])
    D, W = mean_delta[di], mean_wt[di]
    mag = np.array([mag_by[t] for t in toks])[:, None]
    y = np.array([gene_deseq[eid2col[e]] for _, e in rows2])
    uabs = np.array([universal_abs[eid2ucol[e]] for _, e in rows2])
    keep2 = np.isfinite(y) & (y != 0)
    D, W, mag, uabs, y = D[keep2], W[keep2], mag[keep2], uabs[keep2], y[keep2]
    log.info(f"  aligned genes {keep2.sum()}")

    r_wt = cv_r(W, y); r_d = cv_r(D, y); r_dw = cv_r(np.column_stack([D, W, mag]), y)
    log.info(f"\n===== {GENE} DESeq2-target increment diagnostic =====")
    log.info(f"  WT-alone     : {r_wt:+.4f}")
    log.info(f"  delta-alone  : {r_d:+.4f}")
    log.info(f"  delta+WT+mag : {r_dw:+.4f}")
    log.info(f"  delta increment : {r_dw - r_wt:+.4f}")
    yabs = np.abs(y)
    r_sb = cv_r(np.column_stack([uabs, W]), yabs)
    r_sf = cv_r(np.column_stack([uabs, W, D]), yabs)
    log.info(f"  [saliency] baseline[universal,WT] {r_sb:+.4f} -> +delta {r_sf:+.4f} (increment {r_sf-r_sb:+.4f})")

    np.savez_compressed(OUTDIR / f"{GENE}_replogle_deseq2.npz",
                        gene_deseq=gene_deseq, eids=eids,
                        r_wt=r_wt, r_dw=r_dw, incr=r_dw - r_wt,
                        sal_incr=r_sf - r_sb, n_rep=n_gene_rep)
    verdict = "~0: the negative conclusion reproduces on Replogle" if abs(r_dw - r_wt) < 0.03 else "WARNING: increment is significant, re-check"
    log.info(f"  -> delta increment {r_dw-r_wt:+.3f} {verdict}")


if __name__ == "__main__":
    main()
