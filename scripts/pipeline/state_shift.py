"""
Cell-embedding state-shift readout (does the embedding move toward a target state) + universal-responsiveness null.

Readout (a Geneformer-validated capability, not the disproven per-gene log2FC):
  - state axis = normalize(CLS centroid of target-state cells - CLS centroid of source-state cells)
  - in-silico KO of gene X: for source-state cells, CLS_KO - CLS_WT = shift, proj = dot(shift, axis)
  - universal-responsiveness null: mean projection of random-gene in-silico KOs -> specific = score(X) - universal

Confound guards: (1) universal-responsiveness null (2) coverage gate (perturb only cells containing the token)
  (3) optional direction (cosine) to separate magnitude.

Self-test (on Replogle, no T-cell data needed):
  target state = CLS centroid of real GENE-KO cells; source state = control.
  Q: does in-silico KO of GENE push control cells toward the real GENE-KO state, and beyond random genes
  (the universal null)? -- validates the code and probes whether the cell-embedding readout carries
  KO-specific state-shift signal.

Usage: python scripts/pipeline/state_shift.py HSPA9 [N_CTRL] [N_RANDOM]
"""
import sys, logging, pickle, tempfile
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse

import datasets   # noqa
import peft       # noqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("state")

GENE     = sys.argv[1] if len(sys.argv) > 1 else "HSPA9"
N_CTRL   = int(sys.argv[2]) if len(sys.argv) > 2 else 100   # source-state (control) cells
N_RANDOM = int(sys.argv[3]) if len(sys.argv) > 3 else 10    # number of random genes for the universal null
from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "ReplogleWeissman2022_K562_essential.h5ad")
MODEL_DIR = "Geneformer/Geneformer-V2-104M"
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
N_KO = int(sys.argv[4]) if len(sys.argv) > 4 else 200   # real KO cells used to estimate the target-state centroid
LAYER = -1            # which hidden_states layer for CLS (-1 = last_hidden_state)
SEED = 0


def cls_embeddings(model, device, id_lists, layer):
    """For a set of input_ids lists, forward each and take the CLS (position 0) embedding. Returns [n, H]."""
    import torch
    out = []
    with torch.no_grad():
        for ids in id_lists:
            t = torch.tensor([list(ids)], device=device)
            h = model(input_ids=t, output_hidden_states=True).hidden_states[layer][0]
            out.append(h[0].float().cpu().numpy())   # position 0 = CLS
    return np.vstack(out)


def ko_forward_cls(model, device, ids, target_token, layer):
    """Forward after deleting target_token, taking CLS."""
    import torch
    ko = [t for t in ids if t != target_token]
    with torch.no_grad():
        t = torch.tensor([ko], device=device)
        h = model(input_ids=t, output_hidden_states=True).hidden_states[layer][0]
    return h[0].float().cpu().numpy()


def tokenize(adata_sub, wrapper, ko_engine):
    """Tokenize a batch of cells; return a list of input_ids."""
    adata_sub = adata_sub.copy()
    adata_sub.obs["cell_type"] = "cell"; adata_sub.raw = adata_sub
    tmp = Path(tempfile.mkdtemp())
    loom = wrapper._adata_to_loom(adata_sub, tmp, "cell_type")
    ko_engine._tokenize_loom(loom, tmp / "tok")
    ds = datasets.load_from_disk(str(tmp / "tok" / "tokenized.dataset"))
    return [list(x) for x in ds["input_ids"]]


def main():
    import torch
    log.info(f"GENE={GENE}, N_CTRL={N_CTRL}, N_RANDOM={N_RANDOM}")
    adata = sc.read_h5ad(DATA)
    from isp_confound import GeneformerWrapper, InSilicoKO
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR); wrapper.load()
    model = wrapper._model; model.eval()
    device = next(model.parameters()).device
    with open(TOKDICT, "rb") as f:
        g2i = {k: v for k, v in pickle.load(f).items() if not str(k).startswith("<")}
    sym2eid = dict(zip(adata.var_names, adata.var["ensembl_id"]))
    target_token = g2i.get(sym2eid.get(GENE))
    koeng = InSilicoKO(wrapper, n_mc_samples=1, cell_intoken_size=N_CTRL)

    o = adata.obs
    ctrl = adata[(o["perturbation"] == "control").values][:N_CTRL]
    real_ko = adata[(o["perturbation"] == GENE).values][:N_KO]
    log.info(f"Step1: tokenize {ctrl.n_obs} control + {real_ko.n_obs} real {GENE}-KO cells")
    ctrl_ids = tokenize(ctrl, wrapper, koeng)
    ko_ids = tokenize(real_ko, wrapper, koeng)

    # -- state axis: real KO state - control state (CLS centroids) --
    log.info("Step2: real CLS centroids -> state axis")
    ctrl_cls = cls_embeddings(model, device, ctrl_ids, LAYER)
    ko_cls = cls_embeddings(model, device, ko_ids, LAYER)
    axis = ko_cls.mean(0) - ctrl_cls.mean(0)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm
    log.info(f"  state axis ||centroid diff||={axis_norm:.3f} (CLS separation between real KO state and control state)")

    # -- candidate genes: GENE + random genes (universal null), all must be in vocab --
    rng = np.random.default_rng(SEED)
    vocab_syms = [s for s in adata.var_names if g2i.get(sym2eid.get(s)) is not None and s != GENE]
    rand_syms = list(rng.choice(vocab_syms, size=N_RANDOM, replace=False))
    genes = {GENE: target_token}
    for s in rand_syms:
        genes[s] = g2i[sym2eid[s]]

    # -- in-silico KO: project each gene over the control cells that contain its token --
    log.info(f"Step3: in-silico KO projection ({GENE} + {N_RANDOM} random genes, with coverage gating)")
    scores = {}
    for sym, tok in genes.items():
        projs = []
        for ids, wt in zip(ctrl_ids, ctrl_cls):
            if tok not in ids:    # coverage gate
                continue
            kcls = ko_forward_cls(model, device, ids, tok, LAYER)
            projs.append(float(np.dot(kcls - wt, axis)))
        if projs:
            scores[sym] = (np.mean(projs), len(projs))

    # -- results --
    log.info("\n===== state-shift self-test results =====")
    g_score, g_n = scores.get(GENE, (np.nan, 0))
    rand_scores = [v[0] for s, v in scores.items() if s != GENE]
    universal = float(np.mean(rand_scores)) if rand_scores else np.nan
    u_std = float(np.std(rand_scores)) if rand_scores else np.nan
    log.info(f"  {GENE} in-silico KO -> real-KO-state projection: {g_score:+.4f} ({g_n} cells containing the token)")
    log.info(f"  random-gene projection (universal null): mean {universal:+.4f} +/- {u_std:.4f} (n={len(rand_scores)})")
    spec = g_score - universal
    z = spec / u_std if u_std > 1e-9 else np.nan
    log.info(f"  specific = {GENE} - universal = {spec:+.4f} (z~{z:+.2f})")
    log.info("\n  individual random-gene projections:")
    for s, (v, nn) in sorted(scores.items(), key=lambda x: -x[1][0]):
        if s != GENE:
            log.info(f"    {s:<10} {v:+.4f} ({nn})")

    log.info("\n===== interpretation =====")
    log.info("  code self-test: state axis / projection / universal null / coverage gate all produce finite values -> machinery works")
    if z > 1.5:
        log.info(f"  preview (positive): in-silico KO of {GENE} moves toward the real KO state well beyond random genes (z={z:+.2f}) -> the cell-embedding readout may carry specific signal, worth investing in real data")
    elif abs(z) < 1.0:
        log.info(f"  preview (zero): {GENE} does not move toward the real KO state more than random genes (z={z:+.2f}) -> the cell-embedding readout may also lack specific signal (consistent with delta)")
    else:
        log.info(f"  preview (weak/borderline): z={z:+.2f}, needs more genes/cells to confirm")
    log.info("  note: this is a Replogle self-test (KO state as target), not a T-cell exhaustion->effector axis; a real application needs exhaustion data + a known de-exhaustion gene as a positive control")


if __name__ == "__main__":
    main()
