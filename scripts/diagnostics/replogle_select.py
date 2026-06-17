"""
Replogle perturbation selection: pick cross-process + high-coverage + strong-and-robust perturbations
to cache for diagnosis.

Four criteria:
  - strong effect: many significant DEGs (number of genes with |log2FC| > 0.5)
  - robust: split-half reliability (correlation of LFCs computed on each half of the cells) ->
    distinguishes noise from real diversity
  - high coverage: the knocked-out gene's token appears in enough control cells' sequences
    (avoids CCND1-style low-coverage confounding)
  - cross-process: flag the ribosome/translation/proteasome stress cluster (the heterogeneity
    pre-check shows it dominates PC1); avoid it and pick across processes

Outputs a ranked candidate table to pick ~5 from. Run from the project root.
"""
import sys, logging, pickle, tempfile, re
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr

import datasets   # noqa  correct import order
import peft       # noqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("select")

from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "ReplogleWeissman2022_K562_essential.h5ad")
MODEL_DIR = "Geneformer/Geneformer-V2-104M"
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
N_COV_CELLS = 300       # control cells used to estimate coverage
MIN_CELLS = 80          # minimum cells per perturbation (enough for split-half + caching)
DEG_THR = 0.5           # |log2FC| DEG threshold
STRESS = re.compile(r"^(RPL|RPS|MRPL|MRPS|EIF|RACK1|FAU|PSM|POLR1|NOP|NHP2|FBL|DKC1|RRP|UTP|WDR12|BOP1|NCL)")


def main():
    OUT = Path("benchmark_output/deseq2"); OUT.mkdir(parents=True, exist_ok=True)
    log.info("Step 1: load Replogle")
    adata = sc.read_h5ad(DATA)
    o = adata.obs
    sym2eid = dict(zip(adata.var_names, adata.var["ensembl_id"]))

    # -- coverage: tokenize control cells (raw counts, before normalization) --
    log.info(f"Step 2: token coverage (tokenize {N_COV_CELLS} control cells)")
    from isp_confound import GeneformerWrapper, InSilicoKO
    with open(TOKDICT, "rb") as f:
        tok_dict = {k: v for k, v in pickle.load(f).items() if not str(k).startswith("<")}
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR)   # do not load the model, only use the tokenizer
    ko = InSilicoKO(wrapper, n_mc_samples=1, cell_intoken_size=N_COV_CELLS)
    ctrl = adata[(o["perturbation"] == "control").values][:N_COV_CELLS].copy()
    ctrl.obs["cell_type"] = "K562"; ctrl.raw = ctrl
    tmp = Path(tempfile.mkdtemp())
    loom = wrapper._adata_to_loom(ctrl, tmp, "cell_type")
    ko._tokenize_loom(loom, tmp / "tok")
    ds = datasets.load_from_disk(str(tmp / "tok" / "tokenized.dataset"))
    from collections import Counter
    pres = Counter()
    for ids in ds["input_ids"]:
        pres.update(set(ids))
    ncov = len(ds)
    log.info(f"  tokenized {ncov} cells")

    # -- effect + robustness: per-perturbation after normalization --
    log.info("Step 3: effect strength + split-half robustness (normalize_total)")
    sc.pp.normalize_total(adata, target_sum=1e4)
    pert = o["perturbation"].astype(str)
    vc = pert.value_counts()
    cands = [p for p in vc.index if p != "control" and vc[p] >= MIN_CELLS]
    log.info(f"  {len(cands)} candidate perturbations (>={MIN_CELLS} cells)")

    Xc = adata[(pert == "control").values].X
    Xc = Xc.toarray() if sparse.issparse(Xc) else np.asarray(Xc)
    logctrl = np.log2(Xc.mean(0) + 1)
    rng = np.random.default_rng(0)

    rows = []
    for p in cands:
        idx = np.where((pert == p).values)[0]
        Xk = adata.X[idx]; Xk = Xk.toarray() if sparse.issparse(Xk) else np.asarray(Xk)
        lfc = np.log2(Xk.mean(0) + 1) - logctrl
        deg = int((np.abs(lfc) > DEG_THR).sum())
        # split-half reliability
        perm = rng.permutation(len(idx)); h = len(idx) // 2
        l1 = np.log2(Xk[perm[:h]].mean(0) + 1) - logctrl
        l2 = np.log2(Xk[perm[h:]].mean(0) + 1) - logctrl
        # compute reliability only on genes with an effect (avoid all-zero noise dominating)
        sel = (np.abs(lfc) > 0.2)
        rel = pearsonr(l1[sel], l2[sel])[0] if sel.sum() > 20 else np.nan
        eid = sym2eid.get(p)
        tok = tok_dict.get(eid)
        cov = pres.get(tok, 0) / ncov if tok is not None else np.nan
        rows.append({"gene": p, "n_cells": int(len(idx)), "DEG_count": deg,
                     "split_half_r": round(float(rel), 3) if rel == rel else np.nan,
                     "coverage": round(float(cov), 3) if cov == cov else np.nan,
                     "in_vocab": tok is not None,
                     "stress_cluster": bool(STRESS.match(str(p)))})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "replogle_candidates.csv", index=False)

    # -- filter + recommend -------------------------------------
    log.info("Step 4: filter")
    good = df[(df["in_vocab"]) & (~df["stress_cluster"]) &
              (df["coverage"] >= 0.5) & (df["DEG_count"] >= 30) &
              (df["split_half_r"] >= 0.6)].copy()
    good["score"] = good["DEG_count"] * good["split_half_r"] * good["coverage"]
    good = good.sort_values("score", ascending=False)
    log.info(f"  qualified candidates (coverage>=0.5 + DEG>=30 + robust>=0.6 + non-stress-cluster): {len(good)}")
    log.info("\n--- Top 25 qualified candidates (pick ~5 across processes) ---")
    print(good.head(25)[["gene", "n_cells", "DEG_count", "split_half_r", "coverage"]].to_string(index=False))

    log.info("\n--- control: strongest in the stress cluster (should be avoided) ---")
    st = df[df["stress_cluster"] & (df["coverage"] >= 0.5)].nlargest(5, "DEG_count")
    print(st[["gene", "DEG_count", "split_half_r", "coverage"]].to_string(index=False))

    log.info(f"\nfull candidate table: {OUT/'replogle_candidates.csv'}")
    log.info("next: from the top qualified candidates, pick ~5 spanning different biological processes, then path-D cache + DESeq2 increment diagnostic")


if __name__ == "__main__":
    main()
