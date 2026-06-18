# isp-confound-toolkit

**A confound-diagnostic toolkit for judging when to trust foundation-model in-silico perturbation predictions** — with Geneformer as a worked cautionary case.

Single-cell foundation models (Geneformer, scGPT, …) are widely proposed for *in-silico perturbation*: predict a knockout's transcriptional effect from a frozen model by deleting the gene's token and reading the embedding response (the "delta"). Evaluating such predictions is error-prone, because apparent signal is easily manufactured by confounds. This repository provides reusable, confound-aware checks for that evaluation.

> **What this is.** The contribution is the **evaluation toolkit** (`scripts/`). The accompanying empirical finding — that Geneformer's *frozen, native* in-silico knockout delta carries no perturbation-specific signal beyond a gene-identity baseline — is reported as **independent confirmation**, consistent with concurrent reports (Ahlmann-Eltze et al. 2025; Kedzierska et al. 2025; scISP / Liu et al. 2025; scFME / Boylan et al. 2025; Kendiukhov 2026). A methods preprint is in preparation.

## The toolkit

Model-agnostic checks you can apply to your own tokenized single-cell model:

- **path-D** (`scripts/pipeline/`) - a machine-precision re-implementation of the per-cell knockout response, validated against the official `InSilicoPerturber` to Pearson *r* = 1.000000 (max abs error 3e-7). Responses are cached and resampled in pure NumPy, enabling a **controlled coverage-to-estimate** analysis (subsample the contributing cells while holding the model, tokenization, and perturbation operator fixed; coverage improves estimate stability/applicability, not biological specificity).
- **Universal-responsiveness control** — hub / highly expressed genes shift under *any* perturbation; subtract the across-perturbation mean before claiming gene-specific saliency.
- **Tokenization-coverage gate** — a target absent from most cells' top-4096 token sequences cannot be knocked out in silico; gate targets by real coverage.
- **Library-size contamination check** — raw-count log2FC is depth-confounded; use size-factor-normalized pseudobulk (DESeq2/edgeR), never raw-count log2FC.
- **De-circularization matrix** — for cell-embedding state-shift readouts, project the knockout of gene A onto gene B's axis to remove the self-deletion artifact.
- **Heterogeneity pre-check** — a zero-model check that a dataset *contains* specific signal before attributing a negative result to the model.

## Scope — which models do these checks apply to?

This is **not** one universal standard for every perturbation-prediction model. It is a set of confound-aware *evaluation principles* plus a set of *mechanism-specific* checks — be precise about which part applies to your model.

**Designed scope:** frozen, tokenized single-cell foundation models that do in-silico perturbation by deleting a gene's token and reading the embedding response - Geneformer here. For other architectures, use a model-specific re-implementation and an analogous input-coverage or perturbation-applicability gate; do not assume the Geneformer implementation transfers unchanged.

| Check | Applies to |
|---|---|
| **Library-size contamination** | **Any** perturbation-prediction evaluation — it concerns how the *ground truth* is computed (raw-count log2FC is depth-confounded), independent of the model. |
| **Universal-responsiveness control** | **Any** "which genes are affected" / saliency claim — hub / high-expression genes respond to everything; subtract the across-perturbation baseline first. |
| **Held-out increment-over-baseline + heterogeneity pre-check** | **General evaluation hygiene** — never report in-sample *r*; require the model to beat a gene-identity baseline; confirm the data contains specific signal before blaming the model. |
| **Tokenization-coverage gate** | Only **truncated rank-based token** models (Geneformer top-4096, scGPT). |
| **path-D** | Only the **token-deletion + embedding-readout** mechanism (it reproduces that engine). The idea - cache responses for controlled resampling - transfers; the implementation is mechanism-specific. |
| **De-circularization matrix** | Only **cell-embedding state-shift** readouts (projecting a knockout onto a state axis, where self-deletion can manufacture signal). |

**Different model classes** — supervised perturbation predictors (GEARS, scGen), GRN-based methods (CellOracle), or decoders that output an expression vector directly — have no "delete-token, read-embedding" step, so `path-D`, the coverage gate, and the de-circularization matrix do **not** directly apply. The general subset (library-size, universal-responsiveness, held-out hygiene, heterogeneity pre-check) still does.

## Quickstart

Two ways to use the toolkit. Both start from a clone + editable install:

```bash
git clone <repository-url>
cd isp-confound-toolkit
python -m venv .venv && .venv\Scripts\activate     # Unix: source .venv/bin/activate
pip install -e .
```

**Always run scripts from the repository root.**

### A — Reproduce the paper's results (no GPU)

The path-D caches let every figure and increment test run in pure NumPy. Download the cache bundle from the Zenodo DOI (see [Reproducing from cached results](#reproducing-from-cached-results-no-gpu)) and extract so that `benchmark_output/pathd_cache/*.npz` and `benchmark_output/deseq2/*.npz` exist:

```bash
python scripts/diagnostics/confound_check.py        # confound decomposition
python scripts/diagnostics/nonlinear_seal.py        # linear + nonlinear increment (Fig 2B)
python scripts/pipeline/resample_pathd.py CCND1     # coverage curve (Fig 3B)
```

Curated figures and key CSVs are already in `results/` — nothing to download just to look.

### B — Run from scratch / apply to your own model (GPU + Geneformer)

```bash
# 1. install Geneformer (V2-104M) so it imports as `geneformer`  (see "Installation & prerequisites")
# 2. download the scPerturb datasets
python scripts/download_data.py                     # -> ./data  (or set ISP_DATA_ROOT)
# 3. cache path-D responses (uses the GPU), then run the diagnostics
python scripts/pipeline/cache_pathd.py CCND1
python scripts/pipeline/validate_pathd.py           # expect r = 1.000000
python scripts/diagnostics/confound_check.py
```

To apply the checks to a **different** tokenized model (scGPT, scFoundation, …), swap the wrapper / in-silico-KO engine in `isp_confound` for your model. The universal-responsiveness and library-size checks are model-independent; the coverage gate, path-D, and de-circularization matrix are mechanism-specific and need a model-specific re-implementation of the knockout response.

## Repository layout

```
isp_confound/        # supporting utilities (NOT the headline): a thin Geneformer
                     #   access layer + in-silico KO engine used by the scripts
scripts/
  pipeline/          # path-D: cache / validate / resample; state-shift + de-circularization
  diagnostics/       # the checks: token coverage, confound checks, DESeq2 ground truth,
                     #   increment tests, heterogeneity pre-check, figure generation
  benchmark/         # end-to-end Frangieh benchmark
tests/               # unit tests for the supporting utilities
results/             # curated small outputs (figures + key CSVs)
```

## Installation & prerequisites

Python ≥3.10; a CUDA GPU is recommended (developed on PyTorch 2.5 + CUDA 12.1, NVIDIA T400 4 GB). `transformers` is pinned `<5.0` (5.x removes `SpecialTokensMixin`, which Geneformer requires).

### Geneformer (separate, not redistributed here)

Geneformer's source and **V2-104M** weights are **not** included (large; third-party license). Obtain them from the official release and make them importable as `geneformer` (e.g. clone the Hugging Face repo and `pip install -e` it, or place it at `./Geneformer`). The code auto-discovers the token dictionary (`token_dictionary_gc104M.pkl`) inside the installed `geneformer` package.

### Datasets (download separately)

Public, preprocessed h5ad files from **scPerturb** (Peidli et al., *Nat Methods* 2024, https://scperturb.org):

- `FrangiehIzar2021_RNA.h5ad` — melanoma + tumor-infiltrating lymphocytes (Perturb-CITE-seq)
- `ReplogleWeissman2022_K562_essential.h5ad` — genome-scale K562 Perturb-seq (CRISPRi)

Place them under `./data/` (or set `ISP_DATA_ROOT` to a different directory).

## Reproducing from cached results (no GPU)

The path-D per-cell response caches, DESeq2 outputs, figure inputs, and audit tables (approximately 465 MB) are too large to commit. They are available on Zenodo: https://doi.org/10.5281/zenodo.20729460. With those cached intermediates, every figure and increment test reproduces in pure NumPy - no Geneformer forward passes:

```bash
# download the cache bundle from the Zenodo record above, then extract so that
#   benchmark_output/pathd_cache/*.npz  and  benchmark_output/deseq2/*.npz  exist
python scripts/pipeline/resample_pathd.py CCND1        # coverage curve (Fig 3B)
python scripts/diagnostics/confound_check.py           # confound decomposition
python scripts/diagnostics/nonlinear_seal.py           # linear + nonlinear increment (Fig 2B)
python scripts/diagnostics/thin_gate_clean.py          # thin-gate clean-target value
```

Small figures and the key CSVs (`token_coverage.csv`, `CCND1_curve_*.csv`) are tracked in `results/`.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Citation

A methods preprint is in preparation; citation details will be added on posting. The reproduction bundle is available on Zenodo: https://doi.org/10.5281/zenodo.20729460. Please also cite Geneformer (Theodoris et al., *Nature* 2023) and scPerturb (Peidli et al., *Nat Methods* 2024).

## License

MIT — see [LICENSE](LICENSE). Geneformer and the scPerturb datasets are governed by their own licenses and are not redistributed here.
