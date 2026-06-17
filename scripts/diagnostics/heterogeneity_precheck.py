"""
Heterogeneity pre-check (zero-model, ~10 min): is one dataset more heterogeneous than another?

Logic: if a dataset's universal responsiveness correlates very highly with a single perturbation
(e.g. Frangieh vs CCND1 = +0.868), the perturbations may be too homogeneous, capping the specific
signal any method can recover. Only if a second dataset is genuinely more heterogeneous is it worth
caching its embeddings to disambiguate "model fails vs data has no signal". If it is equally
homogeneous, switch datasets rather than burn hours.

Metrics (per dataset, ground-truth only):
  - mean pairwise Pearson correlation of perturbation signed-log2fc profiles (higher = more homogeneous)
  - PC1 variance fraction of the log2fc matrix (higher = dominated by a single shared axis = homogeneous)
  All normalized (normalize_total), high-expression gene subset, apples-to-apples cell threshold.
"""
import sys, logging
import numpy as np
import scanpy as sc
from scipy import sparse
import warnings; warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("hetero")

MIN_CELLS = 50
TOP_GENES = 2000   # high-expression gene subset (denoise, comparable across datasets)


def profile_matrix(adata, name):
    """Return the [n_pert, top_genes] signed log2fc matrix (normalized, high-expression genes)."""
    log.info(f"[{name}] raw shape {adata.shape}, example obs columns: {list(adata.obs.columns)[:8]}")
    if "perturbation" not in adata.obs.columns:
        log.error(f"[{name}] no 'perturbation' column, skipping"); return None

    # Frangieh: restrict to perturbation_2==Control to keep the background consistent; datasets without it use all cells
    if "perturbation_2" in adata.obs.columns and (adata.obs["perturbation_2"] == "Control").any():
        adata = adata[adata.obs["perturbation_2"] == "Control"].copy()
        log.info(f"[{name}] restrict to perturbation_2==Control -> {adata.n_obs} cells")
    else:
        adata = adata.copy()

    pert = adata.obs["perturbation"].astype(str)
    # single-gene perturbations: no combo separator, not control
    def is_single(v):
        return v != "control" and not any(s in v for s in ["+", ",", "_and_"])
    sub_perts = [p for p in pert.unique() if is_single(p)]
    vc = pert.value_counts()
    sub_perts = [p for p in sub_perts if vc[p] >= MIN_CELLS]
    log.info(f"[{name}] single-gene perturbations (>={MIN_CELLS} cells): {len(sub_perts)}")
    if len(sub_perts) < 10:
        log.error(f"[{name}] too few perturbations, skipping"); return None

    sc.pp.normalize_total(adata, target_sum=1e4)

    ctrl = adata[(pert == "control").values]
    Xc = ctrl.X.toarray() if sparse.issparse(ctrl.X) else np.asarray(ctrl.X)
    if Xc.shape[0] < MIN_CELLS:
        log.error(f"[{name}] not enough control cells ({Xc.shape[0]}), skipping"); return None
    mean_ctrl = Xc.mean(0)
    # high-expression gene subset
    top = np.argsort(mean_ctrl)[::-1][:TOP_GENES]
    logctrl = np.log2(mean_ctrl[top] + 1)

    rows = []
    for p in sub_perts:
        Xk = adata[(pert == p).values].X
        Xk = Xk.toarray() if sparse.issparse(Xk) else np.asarray(Xk)
        rows.append(np.log2(Xk[:, top].mean(0) + 1) - logctrl)
    M = np.vstack(rows)   # [n_pert, top_genes]
    log.info(f"[{name}] profile matrix {M.shape}")
    return M


def heterogeneity(M, name):
    """Mean pairwise correlation + PC1 variance fraction."""
    # pairwise Pearson correlation (rows = perturbations)
    Cc = np.corrcoef(M)
    iu = np.triu_indices_from(Cc, k=1)
    mean_pair = float(np.nanmean(Cc[iu]))
    # PC1 variance fraction
    Mc = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    pc1 = float((s[0] ** 2) / (s ** 2).sum())
    log.info(f"[{name}] mean pairwise correlation = {mean_pair:+.3f} (high = homogeneous); PC1 variance = {pc1*100:.1f}% (high = homogeneous)")
    return mean_pair, pc1


def main():
    # optional datasets: python heterogeneity_precheck.py <name> <path> [<name2> <path2> ...]
    # default compares Frangieh + Norman
    if len(sys.argv) >= 3:
        args = sys.argv[1:]
        datasets = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
    else:
        datasets = [("Frangieh", "data/FrangiehIzar2021_RNA.h5ad"),
                    ("Norman", "data/NormanWeissman2019_filtered.h5ad")]
    res = {}
    for name, path in datasets:
        log.info(f"\n{'='*55}\nloading {name}")
        try:
            ad = sc.read_h5ad(path)
            M = profile_matrix(ad, name)
            if M is not None:
                res[name] = heterogeneity(M, name)
        except Exception as e:
            log.error(f"[{name}] failed: {e}")

    log.info("\n" + "=" * 55)
    log.info("conclusion")
    log.info("=" * 55)
    if "Frangieh" in res and "Norman" in res:
        fp, fpc = res["Frangieh"]; npr, npc = res["Norman"]
        log.info(f"  Frangieh: mean pairwise correlation {fp:+.3f}, PC1 {fpc*100:.1f}%")
        log.info(f"  Norman  : mean pairwise correlation {npr:+.3f}, PC1 {npc*100:.1f}%")
        if npr < fp - 0.1 or npc < fpc - 0.1:
            log.info("  -> Norman is clearly more heterogeneous -> worth caching embeddings for a specificity control")
        elif npr > fp - 0.05 and npc > fpc - 0.05:
            log.info("  -> Norman is not more heterogeneous than Frangieh -> cannot disambiguate, switch datasets (e.g. Zhu 2025 CD4 T)")
        else:
            log.info("  -> heterogeneity similar/borderline -> caching Norman has limited value, proceed with caution")


if __name__ == "__main__":
    main()
