# Data and Intermediate Results

This project does not redistribute the scPerturb h5ad files, Geneformer weights, or large path-D caches in Git.

## Public Input Datasets

Download the harmonized scPerturb h5ad files and place them under `data/`:

| Dataset | Expected local path | Use in manuscript | Required contents |
|---|---|---|---|
| Frangieh et al. 2021 melanoma/TIL Perturb-CITE-seq | `data/FrangiehIzar2021_RNA.h5ad` | Frangieh CCND1 worked example, token coverage, library-size contamination, DESeq2 target | raw counts, perturbation labels, sgRNA/replicate metadata, gene symbols and Ensembl IDs |
| Replogle et al. 2022 K562 essential-scale CRISPRi | `data/ReplogleWeissman2022_K562_essential.h5ad` | cross-dataset DESeq2 increment tests, GATA1 state-shift/de-circularization | raw counts, perturbation labels, gemgroup/batch metadata, gene symbols and Ensembl IDs |

Access: see scPerturb (Peidli et al. 2024) and https://scperturb.org.

## Geneformer Assets

The repository expects Geneformer V2-104M to be available locally:

```text
Geneformer/Geneformer-V2-104M/
Geneformer/geneformer/token_dictionary_gc104M.pkl
```

Geneformer source/weights are third-party assets and are not redistributed here. Install Geneformer separately so the `geneformer` Python package is importable.

## Large Intermediate Files

Large generated artifacts are not intended for Git. They will be deposited on Zenodo upon preprint posting, including:

| Artifact class | Example local path | Purpose |
|---|---|---|
| path-D caches | `benchmark_output/pathd_cache/CCND1_cache.npz` | cached per-cell/per-gene responses enabling NumPy-only resampling and readout tests |
| DESeq2 outputs | `benchmark_output/deseq2/*_replogle_deseq2.npz` | primary target and increment-test outputs |
| validation outputs | `benchmark_output/pathd_validation.csv` | path-D vs official engine validation |

The Zenodo DOI will be added upon release.

The small curated figure-input CSVs — `results/data/token_coverage.csv`, `results/data/CCND1_curve_main.csv`, `results/data/CCND1_curve_abl.csv` — are **committed** to this repository (not Zenodo), so Figure 3 reproduces from a clean checkout.

## Recomputing Instead of Downloading Caches

The full path-D cache generation requires running Geneformer forward passes for WT and KO token sequences. A CUDA GPU is recommended; the worked Frangieh CCND1 path-D cache was generated on a 4 GB NVIDIA GPU by batching/subsampling.

Representative commands:

```bash
python scripts/pipeline/cache_pathd.py CCND1 data/FrangiehIzar2021_RNA.h5ad
python scripts/pipeline/validate_pathd.py
python scripts/pipeline/resample_pathd.py CCND1
python scripts/diagnostics/deseq2_groundtruth.py
python scripts/diagnostics/deseq2_replogle.py HSPA9
python scripts/diagnostics/deseq2_replogle.py SUPT6H
python scripts/diagnostics/deseq2_replogle.py CSE1L
python scripts/diagnostics/deseq2_replogle.py DHX15
python scripts/diagnostics/deseq2_replogle.py GATA1
```

See `README_reproduce_figures.md` for panel-level figure provenance and `reported_numbers_source_table.csv` for manuscript-number provenance.
