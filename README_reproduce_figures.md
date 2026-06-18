# Figure Reproduction Audit Trail

This file records panel-level provenance for the three manuscript figures. It is intended for reviewer-facing reproducibility checks, not as a polished user tutorial.

**What is committed vs deposited.** The plotting scripts, the small curated figure-input CSVs (under `results/data/`), and the rendered figures (under `results/figures/`) are committed to this repository. The larger provenance artifacts referenced below as `benchmark_output/...` (run logs, `*.npz` caches, `pathd_validation.csv`) are **not** committed; they are deposited on Zenodo upon preprint posting (see `DATA.md`). Some Figure 1 and Figure 2 panel values are hard-coded in the plotting scripts after being read from those logs (see *Current Reproducibility Gaps*); `reported_numbers_source_table.csv` records the per-number provenance.

## Minimal Commands

Run from the repository root:

```bash
python scripts/diagnostics/make_brief_figure.py
python scripts/diagnostics/make_crossdataset_figure.py
python scripts/diagnostics/make_fig3.py
```

Expected outputs:

| Figure | PNG output | PDF output |
|---|---|---|
| Figure 1 | `results/figures/figure1_cautionary_case.png` | `results/figures/figure1_cautionary_case.pdf` |
| Figure 2 | `results/figures/figure2_crossdataset.png` | `results/figures/figure2_crossdataset.pdf` |
| Figure 3 | `results/figures/figure3_coverage_pathD.png` | `results/figures/figure3_coverage_pathD.pdf` |

## Panel-Level Provenance

Provenance inputs marked `benchmark_output/...` are Zenodo-deposited (not committed); inputs under `results/data/` are committed.

| Panel | Plotting script | Plotting entry point | Primary inputs | Output | Source-table rows |
|---|---|---|---|---|---|
| Figure 1A | `scripts/diagnostics/make_brief_figure.py` | top-level script | `benchmark_output/verify_logs/deseq2_groundtruth_rerun.log`; `benchmark_output/verify_logs/confound_full.log` (Zenodo) | `results/figures/figure1_cautionary_case.png` | R007, R008, R020-R023 |
| Figure 1B | `scripts/diagnostics/make_brief_figure.py` | top-level script | `benchmark_output/verify_logs/probe_CCND1.log`; `benchmark_output/verify_logs/normrecheck.log`; `benchmark_output/verify_logs/path2.log` (Zenodo); `reported_numbers_source_table.csv` | `results/figures/figure1_cautionary_case.png` | R003-R006, R009-R010 |
| Figure 2A | `scripts/diagnostics/make_crossdataset_figure.py` | top-level script | `benchmark_output/verify_logs/deseq2_groundtruth_rerun.log`; `benchmark_output/deseq2/*_replogle_deseq2.npz` (Zenodo) | `results/figures/figure2_crossdataset.png` | R021-R023, R026-R030 |
| Figure 2B | `scripts/diagnostics/make_crossdataset_figure.py` | top-level script | `reported_numbers_source_table.csv`; `benchmark_output/deseq2/*_replogle_deseq2.npz` (Zenodo); `scripts/diagnostics/nonlinear_seal.py` | `results/figures/figure2_crossdataset.png` | R023-R030 |
| Figure 3A | `scripts/diagnostics/make_fig3.py` | top-level script | `results/data/token_coverage.csv` | `results/figures/figure3_coverage_pathD.png` | R044-R046 |
| Figure 3B | `scripts/diagnostics/make_fig3.py` | top-level script | `results/data/CCND1_curve_main.csv`; `results/data/CCND1_curve_abl.csv`; `benchmark_output/pathd_validation.csv` (Zenodo) | `results/figures/figure3_coverage_pathD.png` | R001-R002, R047 |

## Current Reproducibility Gaps

The figures are reproducible from the current repository state, but two improvements would make the audit trail stronger:

1. Some Figure 1 and Figure 2 values are hard-coded in plotting scripts after being copied from logs. The next improvement is to export these values to machine-readable CSV/NPZ result tables and make the plotting scripts read those tables directly.
2. Replogle sign-check values are recorded in the manuscript text, while `*_replogle_deseq2.npz` stores `r_wt`, `r_dw`, `incr`, `sal_incr`, and `n_rep` but not the sign-check. The next rerun of `scripts/diagnostics/deseq2_replogle.py` should export sign-check and control replicate count.

## Primary vs Diagnostic Evidence

Primary evidence for the manuscript conclusion is the held-out increment of `baseline + delta` over the gene-identity baseline on DESeq2 targets, across Frangieh and Replogle, under linear and nonlinear readouts.

Diagnostic-only or exploratory values include raw-count log2FC results, oracle-direction bounds, raw-target thin-gate and rich-probe apparent positives, and the caveated GATA1 cell-state shift. These are included to expose failure modes, not to support the main positive/negative conclusion by themselves. Their roles are encoded in `reported_numbers_source_table.csv` under `claim_role`.
