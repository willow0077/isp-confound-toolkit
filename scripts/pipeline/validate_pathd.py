"""
path-D validation: self-computed per-cell gene cosine vs Geneformer's official stats.csv (per gene).

Reproduces the per-gene cosine of Geneformer's InSilicoPerturber (cls_and_gene, delete):
  layer_to_quant = n_layers - 1 (emb_layer=-1, second-to-last hidden_states)
  WT forward -> drop the perturbed gene's position -> strip CLS/EOS = original_emb
  KO sequence (token deleted) forward -> strip CLS/EOS = perturbation_emb
  CosineSimilarity(dim=2) position-wise -> one per-cell cosine per affected gene

Validation criterion:
  for each affected gene of CD274, self-computed mean cosine vs stats.csv Cosine_sim_mean
  require Pearson r > 0.99 and max absolute error < 0.01
  and check for systematic offset (slope / intercept)
"""

import logging
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import datasets   # noqa  correct import order
import peft       # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("validate")

MODEL_DIR = "Geneformer/Geneformer-V2-104M"
TOK_PATH = "benchmark_output/CD274/tokenized/tokenized.dataset"
STATS_CSV = "benchmark_output/CD274/perturb_output/sample_0/stats.csv"
TARGET_GENE = "CD274"
TARGET_EID = "ENSG00000120217"


def main():
    import torch

    # -- 1. load model + vocabulary -----------------------------
    log.info("Step 1: load the wrapper model")
    from isp_confound import GeneformerWrapper
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR)
    wrapper.load()
    model = wrapper._model
    model.eval()
    device = next(model.parameters()).device
    gene2id = wrapper.get_gene2id()                  # ensembl_id -> token_id
    id2gene = {v: k for k, v in gene2id.items()}     # token_id -> ensembl_id

    target_token = gene2id[TARGET_EID]
    log.info(f"  {TARGET_GENE} token_id = {target_token}")

    # number of layers
    layer_nums = []
    for name, _ in model.named_parameters():
        if "layer." in name:
            layer_nums.append(int(name.split("layer.")[1].split(".")[0]))
    n_layers = max(layer_nums) + 1
    layer_to_quant = n_layers - 1   # emb_layer = -1
    log.info(f"  n_layers={n_layers}, layer_to_quant={layer_to_quant} (second-to-last hidden_states)")

    # CLS / EOS token id (from the token dict)
    import geneformer as gf
    pkg = Path(gf.__file__).parent
    with open(pkg / "token_dictionary_gc104M.pkl", "rb") as f:
        tok_dict = pickle.load(f)
    cls_id = tok_dict.get("<cls>")
    eos_id = tok_dict.get("<eos>")
    log.info(f"  cls_id={cls_id}, eos_id={eos_id}")

    # -- 2. load tokenized cells, keep those containing CD274 ----
    log.info("Step 2: select cells containing the target gene token")
    ds = datasets.load_from_disk(TOK_PATH)
    valid = [i for i in range(len(ds)) if target_token in ds[i]["input_ids"]]
    log.info(f"  of {len(ds)} cells, those containing {TARGET_GENE}: {len(valid)} (should = the perturber log's '# cells with genes_to_perturb')")

    # confirm first/last are CLS/EOS
    ex = ds[valid[0]]["input_ids"]
    log.info(f"  sample input_ids[0]={ex[0]} (CLS?), input_ids[-1]={ex[-1]} (EOS?), len={len(ex)}")

    # -- 3. self-compute per-cell, per-gene cosine --------------
    log.info("Step 3: self-compute per-cell gene cosine")
    from collections import defaultdict
    my_cos = defaultdict(list)   # affected_token -> [cosine per cell]
    cos_fn = torch.nn.CosineSimilarity(dim=2)

    for ci, idx in enumerate(valid):
        input_ids = list(ds[idx]["input_ids"])
        p = input_ids.index(target_token)   # position of the perturbed gene in the full sequence

        ids_t = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            wt_out = model(input_ids=ids_t, output_hidden_states=True)
        wt_full = wt_out.hidden_states[layer_to_quant]          # [1, L, H]
        # drop the perturbed position p, then strip CLS(0)/EOS(-1)
        keep = [j for j in range(wt_full.size(1)) if j != p]
        wt_emb = wt_full[:, keep, :][:, 1:-1, :]               # [1, L-3, H]

        # KO sequence: delete the token at position p
        ko_ids = input_ids[:p] + input_ids[p + 1:]
        ko_t = torch.tensor([ko_ids], device=device)
        with torch.no_grad():
            ko_out = model(input_ids=ko_t, output_hidden_states=True)
        ko_full = ko_out.hidden_states[layer_to_quant]         # [1, L-1, H]
        ko_emb = ko_full[:, 1:-1, :]                           # [1, L-3, H]

        # affected gene list: original sequence minus CLS/EOS/perturbed gene, aligned with emb order
        affected = [t for j, t in enumerate(input_ids)
                    if j not in (0, len(input_ids) - 1) and j != p]

        assert wt_emb.size(1) == ko_emb.size(1) == len(affected), \
            f"dimension mismatch: wt={wt_emb.size(1)} ko={ko_emb.size(1)} aff={len(affected)}"

        cos = cos_fn(ko_emb, wt_emb)[0].cpu().numpy()   # [L-3]
        for tok, c in zip(affected, cos):
            my_cos[tok].append(float(c))

    log.info(f"  self-computed coverage: {len(my_cos)} affected gene tokens")

    # -- 4. compare per gene vs stats.csv -----------------------
    log.info("Step 4: per-gene comparison vs stats.csv")
    stats = pd.read_csv(STATS_CSV).dropna(subset=["Affected_Ensembl_ID"])
    # official Cosine_sim_mean per affected ensembl id
    official = dict(zip(stats["Affected_Ensembl_ID"], stats["Cosine_sim_mean"]))

    rows = []
    for tok, vals in my_cos.items():
        eid = id2gene.get(tok)
        if eid is None or eid not in official:
            continue
        rows.append({
            "affected_eid": eid,
            "my_mean": np.mean(vals),
            "official_mean": official[eid],
            "n_cells": len(vals),
        })
    cmp = pd.DataFrame(rows)
    cmp["abs_err"] = (cmp["my_mean"] - cmp["official_mean"]).abs()

    from scipy.stats import pearsonr
    r, _ = pearsonr(cmp["my_mean"], cmp["official_mean"])
    max_err = cmp["abs_err"].max()
    mean_err = cmp["abs_err"].mean()

    # systematic-offset check: linear fit official = a*my + b
    a, b = np.polyfit(cmp["my_mean"], cmp["official_mean"], 1)

    log.info("\n" + "=" * 60)
    log.info("validation result")
    log.info("=" * 60)
    log.info(f"genes compared: {len(cmp)}")
    log.info(f"Pearson r       : {r:.6f}   (criterion > 0.99)")
    log.info(f"max abs error   : {max_err:.6f} (criterion < 0.01)")
    log.info(f"mean abs error  : {mean_err:.6f}")
    log.info(f"linear fit official = {a:.5f} * my + {b:.6f}  (ideal a=1, b=0)")

    log.info("\n--- 8 genes with the largest error ---")
    print(cmp.nlargest(8, "abs_err")[
        ["affected_eid", "my_mean", "official_mean", "abs_err", "n_cells"]
    ].to_string(index=False))

    passed = (r > 0.99) and (max_err < 0.01)
    log.info("\n" + ("PASS: path-D is usable" if passed
                     else "FAIL: investigate the source of the discrepancy (see linear fit a/b)"))
    cmp.to_csv("benchmark_output/pathd_validation.csv", index=False)
    log.info("per-gene comparison table saved to benchmark_output/pathd_validation.csv")
    return passed


if __name__ == "__main__":
    main()
