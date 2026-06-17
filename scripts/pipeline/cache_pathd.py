"""
path-D cache: for one gene, run a single WT+KO forward pass over the full cell pool
and cache the per-cell, per-gene (cosine magnitude, delta direction projection) to disk.

Cache once (~37 min/gene); afterwards subsampling + resampling + plotting are pure NumPy
(see resample_pathd.py).

Design:
  - cosine (magnitude) uses hidden_states[11] (second-to-last layer, matches the perturber,
    machine-precision validated)
  - direction uses hidden_states[12] (last_hidden_state); delta = KO - WT projected onto the expression axis
  - the expression axis is fixed to the full-pool reference frame (only the coordinate system is
    fixed, not the perturbation answer -> no leakage)
    * estimate the axis on 100 cells and fix it; the main run projects delta with it
    * the main run also accumulates the full-pool axis; afterwards check cos(axis_100, axis_full) > 0.99
      (confirms the "axis is stable" assumption)
  - sign convention: proj > 0 = moves toward the high-expression pole after KO = up-regulated,
    consistent with ground-truth log2FC positive = up

Usage: set the GENE constant and run (CCND1 main experiment / TIMP1 weak-signal control).
"""

import logging
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import datasets   # noqa  correct import order
import peft       # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cache")

# -- Config ----------------------------------------------------
GENE        = sys.argv[1] if len(sys.argv) > 1 else "CCND1"   # gene name from the command line
DATA_PATH   = sys.argv[2] if len(sys.argv) > 2 else "data/FrangiehIzar2021_RNA.h5ad"  # dataset (optional)
MODEL_DIR   = "Geneformer/Geneformer-V2-104M"
OUTPUT_DIR  = Path("benchmark_output/pathd_cache")
N_CTRL      = 600              # number of control cells used to build the valid pool
CONDITION   = "Control"        # perturbation_2 value
N_AXIS      = 100              # number of cells used to estimate the expression axis
N_BOUNDARY  = 50               # number of high/low-expression genes for the expression axis
LAYER_COS   = 11               # cosine (magnitude) layer = second-to-last layer
LAYER_DIR   = 12               # direction layer = last layer (last_hidden_state)
# --------------------------------------------------------------


def compute_ground_truth(adata, ko_gene, condition, gene2id):
    """log2FC of KO cells vs control (positive = up after KO); returns {token_id: log2fc}.
    Note: this is raw-count log2FC, archived in the npz only; the primary DESeq2 target uses DESeq2 (deseq2_groundtruth)."""
    # dataset-aware: Frangieh has a perturbation_2 condition, Replogle does not
    if "perturbation_2" in adata.obs.columns and condition is not None:
        base = (adata.obs["perturbation_2"] == condition)
        mask_ctrl = (adata.obs["perturbation"] == "control") & base
        mask_ko   = (adata.obs["perturbation"] == ko_gene) & base
    else:
        mask_ctrl = adata.obs["perturbation"] == "control"
        mask_ko   = adata.obs["perturbation"] == ko_gene
    a_ctrl, a_ko = adata[mask_ctrl], adata[mask_ko]
    log.info(f"  ground truth: {a_ctrl.n_obs} ctrl cells, {a_ko.n_obs} KO cells")

    X_ctrl, X_ko = a_ctrl.X, a_ko.X
    if sparse.issparse(X_ctrl):
        X_ctrl, X_ko = X_ctrl.toarray(), X_ko.toarray()
    mean_ctrl = X_ctrl.mean(axis=0) + 1.0
    mean_ko   = X_ko.mean(axis=0)   + 1.0
    log2fc = np.log2(mean_ko) - np.log2(mean_ctrl)

    # map ensembl_id -> token_id
    eids = adata.var["ensembl_id"].values
    gt = {}
    for eid, fc in zip(eids, log2fc):
        tok = gene2id.get(eid)
        if tok is not None:
            gt[tok] = float(fc)
    log.info(f"  ground truth covers {len(gt)} in-vocabulary genes, log2FC range [{log2fc.min():.2f}, {log2fc.max():.2f}]")
    return gt


def main():
    import torch
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. load data + model -----------------------------------
    log.info(f"Step 1: load data + model (GENE={GENE})")
    adata = sc.read_h5ad(DATA_PATH)
    from isp_confound import GeneformerWrapper, InSilicoKO
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR)
    wrapper.load()
    model = wrapper._model
    model.eval()
    device = next(model.parameters()).device
    gene2id = wrapper.get_gene2id()
    target_token = gene2id.get(adata.var.loc[GENE, "ensembl_id"]) if GENE in adata.var_names else None
    if target_token is None:
        # fallback: find ensembl from var
        eid = adata.var.loc[GENE, "ensembl_id"]
        target_token = gene2id.get(eid)
    log.info(f"  {GENE} token_id = {target_token}")

    # CLS / EOS token id (for excluding from rich features)
    import geneformer as gf
    with open(Path(gf.__file__).parent / "token_dictionary_gc104M.pkl", "rb") as f:
        _td = pickle.load(f)
    cls_id, eos_id = _td.get("<cls>"), _td.get("<eos>")

    # -- 2. ground truth ----------------------------------------
    log.info("Step 2: compute ground-truth log2FC")
    gt = compute_ground_truth(adata, GENE, CONDITION, gene2id)

    # -- 3. build the valid pool (control cells containing the target token) --
    log.info(f"Step 3: build the valid pool (N_CTRL={N_CTRL})")
    if "perturbation_2" in adata.obs.columns:
        ctrl_mask = (adata.obs["perturbation"] == "control") & (adata.obs["perturbation_2"] == CONDITION)
    else:
        ctrl_mask = adata.obs["perturbation"] == "control"
    a_ctrl = adata[ctrl_mask][:N_CTRL].copy()
    a_ctrl.obs["cell_type"] = "melanocyte"
    a_ctrl.raw = a_ctrl

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    ko = InSilicoKO(wrapper, n_mc_samples=1, cell_intoken_size=a_ctrl.n_obs)
    loom_path = wrapper._adata_to_loom(a_ctrl, tmp, "cell_type")
    ko._tokenize_loom(loom_path, tmp / "tokenized")
    ds = datasets.load_from_disk(str(tmp / "tokenized" / "tokenized.dataset"))

    valid = [i for i in range(len(ds)) if target_token in ds[i]["input_ids"]]
    log.info(f"  of {len(ds)} control cells, the valid pool containing {GENE}: {len(valid)}")
    if len(valid) < 80:
        log.warning(f"  valid pool < 80, subsampling ceiling is limited!")

    # -- 4. estimate the expression axis first (first N_AXIS valid cells, WT-only) --
    log.info(f"Step 4: estimate the expression axis on the first {N_AXIS} valid cells (fixed reference frame)")
    layer_dir = LAYER_DIR
    axis_sum_100 = np.zeros(model.config.hidden_size, dtype=np.float64)
    n_axis_used = 0
    with torch.no_grad():
        for idx in valid[:N_AXIS]:
            ids = list(ds[idx]["input_ids"])
            if len(ids) < 2 * N_BOUNDARY + 2:
                continue
            t = torch.tensor([ids], device=device)
            h = model(input_ids=t, output_hidden_states=True).hidden_states[layer_dir][0].float().cpu().numpy()
            axis_sum_100 += (h[1:N_BOUNDARY + 1].mean(0) - h[-N_BOUNDARY - 1:-1].mean(0)).astype(np.float64)
            n_axis_used += 1
    axis_100 = axis_sum_100 / n_axis_used
    axis_100 /= np.linalg.norm(axis_100)
    log.info(f"  axis fixed (using {n_axis_used} cells)")

    # -- 5. main run: per-cell, per-gene cosine + delta projection --
    log.info(f"Step 5: main run over the full pool of {len(valid)} cells (WT+KO forward)")
    cos_layer = LAYER_COS
    cos_fn = torch.nn.CosineSimilarity(dim=1)

    coo_gene, coo_cellpos, coo_cos, coo_proj = [], [], [], []
    axis_sum_full = np.zeros(model.config.hidden_size, dtype=np.float64)
    n_full = 0

    # rich features: pre-scan all affected tokens, preallocate per-gene mean delta/WT accumulators
    H = model.config.hidden_size
    cls_id_local, eos_id_local = cls_id, eos_id
    exclude = {cls_id_local, eos_id_local, int(target_token)}
    all_tok = set()
    for idx in valid:
        all_tok.update(ds[idx]["input_ids"])
    all_tok = sorted(all_tok - exclude)
    rtok2col = {t: i for i, t in enumerate(all_tok)}
    delta_sum = np.zeros((len(all_tok), H), dtype=np.float64)   # layer-12 delta(KO-WT) accumulator
    wt_sum = np.zeros((len(all_tok), H), dtype=np.float64)      # layer-12 WT emb accumulator (context feature)
    gene_count = np.zeros(len(all_tok), dtype=np.int64)
    log.info(f"  rich features: {len(all_tok)} affected tokens, mean delta/WT accumulators 2x{len(all_tok)}x{H}")

    import time
    t0 = time.time()
    with torch.no_grad():
        for cell_pos, idx in enumerate(valid):
            ids = list(ds[idx]["input_ids"])
            p = ids.index(target_token)

            wt_t = torch.tensor([ids], device=device)
            wt_out = model(input_ids=wt_t, output_hidden_states=True)
            wt11 = wt_out.hidden_states[cos_layer][0]      # [L,H]
            wt12 = wt_out.hidden_states[layer_dir][0]      # [L,H]

            # full-pool axis accumulation (WT layer 12, full sequence)
            if len(ids) >= 2 * N_BOUNDARY + 2:
                w12 = wt12.float().cpu().numpy()
                axis_sum_full += (w12[1:N_BOUNDARY + 1].mean(0) - w12[-N_BOUNDARY - 1:-1].mean(0)).astype(np.float64)
                n_full += 1

            ko_ids = ids[:p] + ids[p + 1:]
            ko_t = torch.tensor([ko_ids], device=device)
            ko_out = model(input_ids=ko_t, output_hidden_states=True)
            ko11 = ko_out.hidden_states[cos_layer][0]
            ko12 = ko_out.hidden_states[layer_dir][0]

            # align: drop position p from the original sequence, then strip CLS/EOS
            keep = [j for j in range(len(ids)) if j != p]
            wt11_a = wt11[keep][1:-1]    # [n_aff,H]
            wt12_a = wt12[keep][1:-1]
            ko11_a = ko11[1:-1]
            ko12_a = ko12[1:-1]
            affected = [ids[j] for j in keep[1:-1]]
            n_aff = len(affected)
            assert wt11_a.shape[0] == ko11_a.shape[0] == n_aff

            # magnitude: cosine (layer 11)
            cos = cos_fn(ko11_a, wt11_a).float().cpu().numpy()   # [n_aff]
            # direction: delta (layer 12) projected onto the fixed axis
            delta = (ko12_a - wt12_a).float().cpu().numpy()       # [n_aff,H]
            wt12_np = wt12_a.float().cpu().numpy()                 # [n_aff,H]
            proj = delta @ axis_100                               # [n_aff]

            coo_gene.append(np.asarray(affected, dtype=np.int32))
            coo_cellpos.append(np.full(n_aff, cell_pos, dtype=np.int16))
            coo_cos.append(cos.astype(np.float32))
            coo_proj.append(proj.astype(np.float32))

            # rich feature accumulation: per-gene mean delta (layer 12) + mean WT (layer 12)
            cols = np.fromiter((rtok2col[t] for t in affected), dtype=np.int64, count=n_aff)
            np.add.at(delta_sum, cols, delta.astype(np.float64))
            np.add.at(wt_sum, cols, wt12_np.astype(np.float64))
            np.add.at(gene_count, cols, 1)

            if (cell_pos + 1) % 50 == 0:
                dt = time.time() - t0
                eta = dt / (cell_pos + 1) * (len(valid) - cell_pos - 1)
                log.info(f"  {cell_pos+1}/{len(valid)} cells, elapsed {dt/60:.1f} min, ETA {eta/60:.1f} min")

    # -- 6. axis stability check --------------------------------
    axis_full = axis_sum_full / n_full
    axis_full /= np.linalg.norm(axis_full)
    axis_check = float(np.dot(axis_100, axis_full))
    log.info(f"\nStep 6: axis stability check cos(axis_100, axis_full) = {axis_check:.5f}")
    if axis_check > 0.99:
        log.info("  OK: axis stable (>0.99), the fixed-axis assumption holds")
    else:
        log.warning(f"  WARNING: axis not stable enough (<0.99); consider increasing N_AXIS and re-running")

    # -- 7. save ------------------------------------------------
    gene_arr = np.concatenate(coo_gene)
    cellpos_arr = np.concatenate(coo_cellpos)
    cos_arr = np.concatenate(coo_cos)
    proj_arr = np.concatenate(coo_proj)

    gt_tokens = np.array(list(gt.keys()), dtype=np.int32)
    gt_fc = np.array(list(gt.values()), dtype=np.float32)

    # rich features: per-gene means (genes seen in at least 1 cell)
    seen = gene_count > 0
    rich_tokens = np.asarray(all_tok, dtype=np.int32)[seen]
    mean_delta = (delta_sum[seen] / gene_count[seen, None]).astype(np.float32)   # [G,H]
    mean_wt = (wt_sum[seen] / gene_count[seen, None]).astype(np.float32)         # [G,H]

    out = OUTPUT_DIR / f"{GENE}_cache.npz"
    np.savez_compressed(
        out,
        gene=gene_arr, cellpos=cellpos_arr, cos=cos_arr, proj=proj_arr,
        gt_tokens=gt_tokens, gt_fc=gt_fc,
        target_token=np.int32(target_token), n_valid=np.int32(len(valid)),
        axis_check=np.float32(axis_check), gene_name=np.array([GENE]),
        rich_tokens=rich_tokens, mean_delta=mean_delta, mean_wt=mean_wt,
    )
    log.info(f"\ncache saved: {out} ({len(gene_arr):,} cell-gene records, {len(valid)} cells)")
    log.info(f"  rich features: mean_delta/mean_wt each {mean_delta.shape}")
    log.info(f"file size: {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
