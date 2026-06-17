"""
Frangieh 2021 benchmark: in-silico KO prediction vs measured differential expression (Pearson r).

Design:
  - Ground truth: log2FC of KO cells vs control cells (same perturbation_2 condition)
  - Prediction: signed_shift = mean_shift x (+1 for "up" / -1 for "down")
  - Metric: Pearson r and Spearman r (per gene, across all predicted genes)

Pilot stage runs only 3 immune-related genes: JAK2, CD274 (PD-L1), B2M.

Run from the project root.
"""

import logging
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy import sparse

# Correct import order
import datasets   # noqa
import peft       # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("benchmark")

# -- Config ----------------------------------------------------
from isp_confound.config import DATA_ROOT; DATA_PATH = str(DATA_ROOT / "FrangiehIzar2021_RNA.h5ad")
MODEL_DIR   = "Geneformer/Geneformer-V2-104M"
OUTPUT_DIR  = Path("benchmark_output")
PILOT_GENES = ["JAK2", "CD274", "B2M"]
N_CTRL      = 100   # number of control cells used for prediction (caps runtime)
N_MC        = 2     # number of Monte Carlo samples
CONDITION   = "Control"   # perturbation_2 value: use cells without co-culture / IFN-gamma for a clean contrast
# --------------------------------------------------------------


def compute_ground_truth(adata, ko_gene: str, condition: str) -> pd.DataFrame:
    """
    Compute log2 fold change of the KO gene vs control (measured differential expression).
    Uses only perturbation_2 == condition cells to keep the background consistent.
    """
    mask_ctrl = (adata.obs["perturbation"] == "control") & (adata.obs["perturbation_2"] == condition)
    mask_ko   = (adata.obs["perturbation"] == ko_gene)   & (adata.obs["perturbation_2"] == condition)

    adata_ctrl = adata[mask_ctrl]
    adata_ko   = adata[mask_ko]

    if adata_ctrl.n_obs == 0 or adata_ko.n_obs == 0:
        return pd.DataFrame()

    # Mean expression per gene (raw counts, with a pseudocount to avoid log(0))
    X_ctrl = adata_ctrl.X
    X_ko   = adata_ko.X
    if sparse.issparse(X_ctrl):
        X_ctrl = X_ctrl.toarray()
        X_ko   = X_ko.toarray()

    mean_ctrl = X_ctrl.mean(axis=0) + 1.0   # pseudocount
    mean_ko   = X_ko.mean(axis=0)   + 1.0

    log2fc = np.log2(mean_ko) - np.log2(mean_ctrl)

    df = pd.DataFrame({
        "gene":    adata.var_names.tolist(),
        "log2fc":  log2fc,
        "mean_ctrl": mean_ctrl - 1.0,
        "mean_ko":   mean_ko   - 1.0,
    })
    log.info(f"  {ko_gene} ground truth: {adata_ctrl.n_obs} ctrl cells, {adata_ko.n_obs} KO cells")
    log.info(f"  log2FC range: [{log2fc.min():.3f}, {log2fc.max():.3f}]")
    return df


def run_prediction(adata_ctrl, ko_gene: str, wrapper, ko, output_dir: Path) -> pd.DataFrame:
    """
    Run the in-silico KO prediction; return a DataFrame with signed_shift.
    signed_shift = mean_shift if direction == "up" else -mean_shift
    """
    gene_dir = output_dir / ko_gene
    gene_dir.mkdir(parents=True, exist_ok=True)

    result = ko.predict(
        adata_ctrl,
        target_gene=ko_gene,
        cell_type=None,
        cell_type_key="perturbation_2",
        output_dir=gene_dir,
    )

    df = result.to_dataframe().copy()
    # Convert to a signed prediction value
    df["signed_shift"] = df["mean_shift"] * df["direction"].map({"up": 1, "down": -1}).fillna(0)
    return df


def compute_correlation(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """
    Merge ground truth and predictions; compute Pearson r and Spearman r.
    Keep only genes that appear in the predictions.
    """
    merged = gt.merge(pred[["gene", "signed_shift", "mean_shift", "direction"]],
                      on="gene", how="inner")
    if len(merged) < 10:
        return {"n_genes": len(merged), "pearson_r": np.nan, "spearman_r": np.nan}

    # Filter very low-expression genes (optional: keep only genes with mean > 0.1 counts)
    expr_mask = (merged["mean_ctrl"] > 0.1) | (merged["mean_ko"] > 0.1)
    merged = merged[expr_mask]

    pr, pp = pearsonr(merged["log2fc"], merged["signed_shift"])
    sr, sp = spearmanr(merged["log2fc"], merged["signed_shift"])

    # Direction accuracy: fraction where signed_shift and log2fc share a sign
    correct_dir = ((merged["log2fc"] * merged["signed_shift"]) > 0).mean()

    return {
        "n_genes":      len(merged),
        "pearson_r":    round(float(pr), 4),
        "pearson_p":    round(float(pp), 4),
        "spearman_r":   round(float(sr), 4),
        "direction_acc": round(float(correct_dir), 4),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- Step 1: load data --------------------------------------
    log.info("Step 1: load Frangieh data")
    adata = sc.read_h5ad(DATA_PATH)
    log.info(f"  shape: {adata.shape}")

    # -- Step 2: prepare the control-cell subset ----------------
    log.info(f"Step 2: extract {CONDITION}-condition control cells (up to {N_CTRL})")
    ctrl_mask = (adata.obs["perturbation"] == "control") & (adata.obs["perturbation_2"] == CONDITION)
    adata_ctrl_full = adata[ctrl_mask]
    n_ctrl = min(N_CTRL, adata_ctrl_full.n_obs)
    adata_ctrl = adata_ctrl_full[:n_ctrl].copy()
    # Geneformer needs a cell_type column (any label works)
    adata_ctrl.obs["cell_type"] = "melanocyte"
    # Ensure raw counts are in .X (Frangieh is already integer raw counts)
    adata_ctrl.raw = adata_ctrl
    log.info(f"  using {adata_ctrl.n_obs} control cells")

    # -- Step 3: load the model ---------------------------------
    log.info("Step 3: load the Geneformer model")
    from isp_confound import GeneformerWrapper, InSilicoKO
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR)
    wrapper.load()
    ko = InSilicoKO(wrapper, n_mc_samples=N_MC, cell_intoken_size=n_ctrl)
    log.info(f"  vocabulary size: {len(wrapper.get_gene2id())} genes")

    # -- Step 4: run the benchmark for each pilot gene ----------
    results = {}
    for gene in PILOT_GENES:
        log.info(f"\n{'='*50}")
        log.info(f"Benchmark gene: {gene}")

        # Ground truth
        gt = compute_ground_truth(adata, gene, CONDITION)
        if gt.empty:
            log.warning(f"  {gene}: not enough cells, skipping")
            continue

        # Prediction
        try:
            pred = run_prediction(adata_ctrl, gene, wrapper, ko, OUTPUT_DIR)
            log.info(f"  prediction done: {len(pred)} genes")
        except Exception as e:
            log.error(f"  {gene} prediction failed: {e}")
            continue

        # Correlation
        corr = compute_correlation(gt, pred)
        results[gene] = corr
        log.info(f"  Pearson r = {corr['pearson_r']}, Spearman r = {corr['spearman_r']}")
        log.info(f"  Direction accuracy = {corr['direction_acc']}, n_genes = {corr['n_genes']}")

    # -- Step 5: summary ----------------------------------------
    log.info("\n" + "="*60)
    log.info("BENCHMARK summary")
    log.info("="*60)
    summary = pd.DataFrame(results).T
    print(summary.to_string())

    summary_path = OUTPUT_DIR / "benchmark_summary.csv"
    summary.to_csv(summary_path)
    log.info(f"\nResults saved to {summary_path}")

    return results


if __name__ == "__main__":
    main()
