"""
De-circularization state-shift test: cross-gene projection matrix M[A][B] = projection of the in-silico
KO of gene A onto the real KO state of gene B.

De-circularization logic:
  - diagonal M[A][A]: contains the self-deletion component ("the token defining the axis was deleted")
  - off-diagonal M[A][B] (A != B): A's token was not deleted into B's axis -> no circularity, pure
    downstream / pathway signal
Interpretation:
  - mitochondrial cluster (HSPA9/PHB/PHB2) should project positively onto each other (same-pathway
    downstream alignment); cross-pathway (-> GATA1/CSE1L/SUPT6H) should be ~0 (specificity)
  - same-pathway off-diagonal positive and cross-pathway zero -> specific signal survives de-circularization
  - only the diagonal positive and off-diagonals all zero -> mostly a self-deletion artifact

Usage: python scripts/pipeline/state_shift_matrix.py
"""
import sys, logging, pickle, tempfile
from pathlib import Path
import numpy as np
import scanpy as sc

import datasets   # noqa
import peft       # noqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("matrix")

from isp_confound.config import DATA_ROOT; DATA = str(DATA_ROOT / "ReplogleWeissman2022_K562_essential.h5ad")
MODEL_DIR = "Geneformer/Geneformer-V2-104M"
from isp_confound import get_geneformer_token_dict_path
TOKDICT = get_geneformer_token_dict_path()
# pathway structure: mitochondrial cluster (should align with each other) + unrelated (specificity control)
GENES = ["HSPA9", "PHB", "PHB2", "GATA1", "CSE1L", "SUPT6H"]
MITO = {"HSPA9", "PHB", "PHB2"}
N_CTRL = 100
N_KO = 100
N_RANDOM = 6
LAYER = -1
SEED = 0


def main():
    import torch
    adata = sc.read_h5ad(DATA)
    from isp_confound import GeneformerWrapper, InSilicoKO
    wrapper = GeneformerWrapper(model_dir=MODEL_DIR); wrapper.load()
    model = wrapper._model; model.eval()
    device = next(model.parameters()).device
    with open(TOKDICT, "rb") as f:
        g2i = {k: v for k, v in pickle.load(f).items() if not str(k).startswith("<")}
    sym2eid = dict(zip(adata.var_names, adata.var["ensembl_id"]))
    tok_of = {g: g2i.get(sym2eid.get(g)) for g in GENES}
    o = adata.obs
    koeng = InSilicoKO(wrapper, n_mc_samples=1, cell_intoken_size=N_CTRL)

    def tokenize(sub):
        sub = sub.copy(); sub.obs["cell_type"] = "cell"; sub.raw = sub
        tmp = Path(tempfile.mkdtemp())
        loom = wrapper._adata_to_loom(sub, tmp, "cell_type")
        koeng._tokenize_loom(loom, tmp / "tok")
        return [list(x) for x in datasets.load_from_disk(str(tmp / "tok" / "tokenized.dataset"))["input_ids"]]

    def cls(ids_list):
        out = []
        with torch.no_grad():
            for ids in ids_list:
                h = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states[LAYER][0]
                out.append(h[0].float().cpu().numpy())
        return np.vstack(out)

    def ko_cls_one(ids, tok):
        ko = [t for t in ids if t != tok]
        with torch.no_grad():
            h = model(input_ids=torch.tensor([ko], device=device), output_hidden_states=True).hidden_states[LAYER][0]
        return h[0].float().cpu().numpy()

    # -- control + each gene's real KO -> axis --
    log.info("Step1: tokenize control + each gene's real KO cells")
    ctrl = adata[(o["perturbation"] == "control").values][:N_CTRL]
    ctrl_ids = tokenize(ctrl)
    ctrl_cls = cls(ctrl_ids)
    ctrl_centroid = ctrl_cls.mean(0)
    axes = {}
    for g in GENES:
        ko = adata[(o["perturbation"] == g).values][:N_KO]
        kids = tokenize(ko)
        ax = cls(kids).mean(0) - ctrl_centroid
        axes[g] = ax / np.linalg.norm(ax)
        log.info(f"  {g} axis ready ({ko.n_obs} real KO)")

    # -- random genes (null rows) --
    rng = np.random.default_rng(SEED)
    vocab = [s for s in adata.var_names if g2i.get(sym2eid.get(s)) is not None and s not in GENES]
    rand = list(rng.choice(vocab, size=N_RANDOM, replace=False))
    for r in rand:
        tok_of[r] = g2i[sym2eid[r]]

    # -- matrix M[A][B] --
    log.info("Step2: in-silico KO each gene -> project onto each axis")
    rows = GENES + rand
    M = {}
    for A in rows:
        tokA = tok_of[A]
        projs = {B: [] for B in GENES}
        for ids, wt in zip(ctrl_ids, ctrl_cls):
            if tokA not in ids:
                continue
            shift = ko_cls_one(ids, tokA) - wt
            for B in GENES:
                projs[B].append(float(np.dot(shift, axes[B])))
        M[A] = {B: (np.mean(v) if v else np.nan) for B, v in projs.items()}
        M[A]["_n"] = len(projs[GENES[0]])

    # -- print the matrix --
    log.info("\n===== projection matrix M[perturbation A][axis B] (x1000) =====")
    hdr = "pert\\axis " + "".join(f"{b[:7]:>9}" for b in GENES) + "   n"
    log.info(hdr)
    for A in rows:
        line = f"{A[:9]:<9}" + "".join(f"{M[A][b]*1000:>9.1f}" if M[A][b] == M[A][b] else f"{'NA':>9}" for b in GENES)
        tag = ""
        if A in MITO: tag = " [mito]"
        elif A in GENES: tag = " [other]"
        else: tag = " [rand]"
        log.info(line + f"   {M[A].get('_n','')}{tag}")

    # -- interpretation --
    null_rows = rand
    log.info("\n===== interpretation (de-circularization: look at the off-diagonal) =====")
    # mitochondrial-block off-diagonal mean (A,B both mito and A!=B) vs cross-pathway off-diagonal vs random null
    mito_off = [M[a][b] for a in MITO for b in MITO if a != b and M[a][b]==M[a][b]]
    cross = [M[a][b] for a in MITO for b in (set(GENES)-MITO) if M[a][b]==M[a][b]]
    null_to_mito = [M[a][b] for a in null_rows for b in MITO if M[a][b]==M[a][b]]
    diag = [M[a][a] for a in GENES if M[a][a]==M[a][a]]
    log.info(f"  diagonal (with circularity) mean       : {np.mean(diag)*1000:+.1f} x1e-3")
    log.info(f"  mito-block off-diagonal (no circ, same pathway): {np.mean(mito_off)*1000:+.1f} x1e-3")
    log.info(f"  cross-pathway off-diagonal (specificity ~0)    : {np.mean(cross)*1000:+.1f} x1e-3")
    log.info(f"  random gene -> mito axis (null)         : {np.mean(null_to_mito)*1000:+.1f} +/- {np.std(null_to_mito)*1000:.1f} x1e-3")
    z = (np.mean(mito_off)-np.mean(null_to_mito))/ (np.std(null_to_mito)+1e-9)
    log.info(f"  -> mito-block off-diagonal vs null: z~{z:+.2f}")
    if np.mean(mito_off) > np.mean(null_to_mito) + 2*np.std(null_to_mito) and np.mean(mito_off) > abs(np.mean(cross)):
        log.info("  -> same-pathway still aligns after de-circularization, cross-pathway weak -> the cell-embedding readout captures *pathway-specific* state-shift (not a self-deletion artifact)")
    elif abs(np.mean(mito_off)) < 2*np.std(null_to_mito):
        log.info("  -> off-diagonal collapses to the null -> mostly a self-deletion artifact, no signal after de-circularization")
    else:
        log.info("  -> borderline, needs more pathways/genes to confirm")
    log.info("  note: this is still a K562 KO-state proxy, not a T-cell effector state; a real application needs T-cell data. Signal present = worth investing, not the same as the application succeeding.")


if __name__ == "__main__":
    main()
