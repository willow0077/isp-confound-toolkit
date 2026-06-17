"""
Real tokenization-coverage statistics (not the raw-count approximation).

Purpose:
  Run real Geneformer tokenization on Frangieh control cells and count, for each candidate KO gene,
  in how many cells' input_ids its token_id appears.
  Used to (1) choose the subsampling main-experiment gene and (2) choose cross-gene scan genes.

Key points:
  - use control cells (perturbation == control & perturbation_2 == Control), consistent with the benchmark
  - count token appearances in input_ids, not adata.X expression
  - Geneformer orders genes by normalized expression, not raw count, so the real token sequence is required
"""

import logging
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

import datasets   # noqa  correct import order
import peft       # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("coverage")

from isp_confound.config import DATA_ROOT; DATA_PATH = str(DATA_ROOT / "FrangiehIzar2021_RNA.h5ad")
MODEL_DIR = "Geneformer/Geneformer-V2-104M"
OUTPUT_DIR = Path("benchmark_output")
N_CTRL = 300          # number of control cells to tokenize (for statistics; more is more stable)
CONDITION = "Control"

# whitelist of signaling-pathway genes (immune / JAK-STAT / interferon / NF-kB / MAPK / antigen presentation, etc.)
SIGNAL_PATHWAY_GENES = {
    # JAK-STAT / interferon
    "JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4",
    "STAT5A", "STAT5B", "STAT6", "SOCS1", "SOCS2", "SOCS3", "IRF1", "IRF2",
    "IRF3", "IRF7", "IRF9", "IFNGR1", "IFNGR2", "IFNAR1", "IFNAR2",
    # antigen presentation / immune evasion
    "B2M", "TAP1", "TAP2", "TAPBP", "HLA-A", "HLA-B", "HLA-C", "NLRC5",
    "CD274", "PDCD1LG2", "CIITA",
    # NF-kB
    "NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "IKBKB", "CHUK",
    "TNFAIP3", "TRAF2", "TRAF3", "MAP3K7",
    # MAPK / PI3K
    "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "BRAF", "RAF1", "KRAS", "NRAS",
    "PIK3CA", "PTEN", "AKT1", "MTOR",
    # Wnt / other core signaling
    "CTNNB1", "MYC", "TP53", "CDKN1A", "CDKN2A",
    # chemokine / cytokine receptors
    "TLR2", "TLR4", "OSMR", "IL6R", "PIM1",
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- 1. load control cells ----------------------------------
    log.info("Step 1: load Frangieh control cells")
    adata = sc.read_h5ad(DATA_PATH)
    ctrl_mask = (adata.obs["perturbation"] == "control") & (
        adata.obs["perturbation_2"] == CONDITION
    )
    adata_ctrl = adata[ctrl_mask][:N_CTRL].copy()
    adata_ctrl.obs["cell_type"] = "melanocyte"
    adata_ctrl.raw = adata_ctrl
    log.info(f"  using {adata_ctrl.n_obs} control cells")

    # -- 2. determine candidate KO genes (Frangieh single-gene KO, in vocab) --
    log.info("Step 2: determine candidate KO genes")
    single_ko = adata.obs[adata.obs["nperts"] == 1]
    ko_genes = single_ko["perturbation"].value_counts()  # gene -> n_ko_cells
    log.info(f"  number of Frangieh single-gene KO conditions: {len(ko_genes)}")

    sym2eid = dict(zip(adata.var_names, adata.var["ensembl_id"]))

    # -- 3. tokenize control cells ------------------------------
    log.info("Step 3: real tokenization (reusing the wrapper pipeline)")
    from isp_confound import GeneformerWrapper, InSilicoKO
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR)
    wrapper.load()
    gene2id = wrapper.get_gene2id()   # ensembl_id -> token_id
    ko = InSilicoKO(wrapper, n_mc_samples=1, cell_intoken_size=adata_ctrl.n_obs)

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    loom_path = wrapper._adata_to_loom(adata_ctrl, tmp, "cell_type")
    tok_dir = tmp / "tokenized"
    ko._tokenize_loom(loom_path, tok_dir)

    ds = datasets.load_from_disk(str(tok_dir / "tokenized.dataset"))
    log.info(f"  number of tokenized cells: {len(ds)}")
    seq_lens = [len(x) for x in ds["input_ids"]]
    log.info(f"  sequence length: min={min(seq_lens)}, max={max(seq_lens)}, median={int(np.median(seq_lens))}")

    # -- 4. count in how many cells each token_id appears -------
    log.info("Step 4: count input_ids token appearances")
    from collections import Counter
    cell_token_sets = [set(ids) for ids in ds["input_ids"]]
    presence = Counter()
    for s in cell_token_sets:
        presence.update(s)
    n_cells = len(cell_token_sets)

    # -- 5. build the coverage table ----------------------------
    log.info("Step 5: build the coverage table")
    rows = []
    for gene, n_ko in ko_genes.items():
        eid = sym2eid.get(gene, None)
        if eid is None or eid not in gene2id:
            continue  # not in vocab
        tok = gene2id[eid]
        n_present = presence.get(tok, 0)
        rows.append({
            "gene": gene,
            "ensembl_id": eid,
            "token_coverage": round(n_present / n_cells, 4),
            "n_cells_with_token": n_present,
            "is_signal_pathway": gene in SIGNAL_PATHWAY_GENES,
            "n_ko_cells": int(n_ko),
        })

    df = pd.DataFrame(rows).sort_values("token_coverage", ascending=False).reset_index(drop=True)
    out_path = OUTPUT_DIR / "token_coverage.csv"
    df.to_csv(out_path, index=False)

    # -- 6. report ----------------------------------------------
    log.info("\n" + "=" * 70)
    log.info(f"real token coverage table ({n_cells} control cells)")
    log.info("=" * 70)
    log.info(f"in-vocabulary KO genes: {len(df)} / {len(ko_genes)}")
    log.info(f"coverage median: {df['token_coverage'].median():.3f}, mean: {df['token_coverage'].mean():.3f}")

    log.info("\n--- top 15 by coverage ---")
    print(df.head(15).to_string(index=False))

    log.info("\n--- bottom 10 by coverage ---")
    print(df.tail(10).to_string(index=False))

    log.info("\n--- signaling-pathway genes (sorted by coverage) ---")
    sig = df[df["is_signal_pathway"]]
    print(sig.to_string(index=False))

    # distribution histogram
    bins = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.01]
    labels = ["0-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-100%"]
    df["bin"] = pd.cut(df["token_coverage"], bins=bins, labels=labels, right=False)
    log.info("\n--- coverage distribution ---")
    print(df["bin"].value_counts().reindex(labels).to_string())

    log.info(f"\nfull table saved to {out_path}")
    return df


if __name__ == "__main__":
    main()
