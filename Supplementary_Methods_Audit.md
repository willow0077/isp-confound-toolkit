# Supplementary Methods Audit Notes

This file is a reviewer-defense companion to the manuscript. It records which analyses are primary evidence and which analyses are exploratory diagnostics used to expose failure modes.

## Primary Evidence

The primary empirical claim is narrow:

Geneformer's frozen, native in-silico knockout embedding delta does not add perturbation-specific signal beyond a gene-identity baseline under the evaluated readouts.

Load-bearing evidence:

| Evidence class | Dataset | Target | Readout | Criterion |
|---|---|---|---|---|
| Frangieh DESeq2 increment test | Frangieh melanoma/TIL | pseudobulk DESeq2 log2FC | ridge | held-out baseline+delta increment over WT gene-identity baseline |
| Frangieh nonlinear increment test | Frangieh melanoma/TIL | pseudobulk DESeq2 log2FC | MLP and gradient-boosted trees | same-model-class held-out increment over WT baseline |
| Replogle cross-dataset increment test | Replogle K562 CRISPRi | pseudobulk DESeq2 log2FC | ridge | held-out baseline+delta increment over WT gene-identity baseline |
| Heterogeneity pre-check | Replogle vs Frangieh | perturbation profiles | split-half and pairwise profile correlation | confirms data contain perturbation-specific signal |

The primary comparison is always:

```text
baseline feature set vs the same baseline feature set + embedding delta
```

A feature set predicting the target is not sufficient evidence; the delta must improve held-out prediction beyond gene identity.

## Exploratory Diagnostics Versus Primary Evidence

| Analysis | Role | Why it is not primary evidence |
|---|---|---|
| Raw-count log2FC readouts | diagnostic_only | raw-count log2FC is intentionally used to expose library-size contamination |
| Oracle-direction upper bound | diagnostic_only | sign flipping makes the apparent positive value a construction artifact |
| Thin [magnitude, projection] readout on raw target | exploratory_only | apparent signal collapses on the primary DESeq2 target |
| Rich 768D delta probe on exploratory target | exploratory_only | WT gene-identity baseline reaches the same value; the delta has no increment |
| Affected-gene saliency before universal-responsiveness control | diagnostic_only | saliency is dominated by genes broadly responsive across perturbations |
| GATA1 cell-state shift | caveated_result | underpowered per-gene DESeq2 evidence and state-shift readout fails de-circularization |
| Coverage-to-estimate curve | diagnostic_only | coverage controls estimate stability and applicability, not biological specificity |

## Path-D Validation

Path-D re-implements Geneformer's native per-cell gene cosine-similarity response and validates against the official `InSilicoPerturber`.

Audit fields:

| Item | Value / source |
|---|---|
| Validation output | `benchmark_output/pathd_validation.csv` |
| Script | `scripts/pipeline/validate_pathd.py` |
| Pearson r | 1.000000 |
| Maximum absolute error | 2.98e-7, reported as 3e-7 |
| Role | documents that downstream resampling manipulates coverage without changing the model, tokenization, or perturbation operator |

## DESeq2 Target Construction

Primary log2FC targets use pseudobulk aggregation and size-factor-normalized count-based differential expression.

Audit fields:

| Item | Frangieh | Replogle |
|---|---|---|
| Biological replicate definition | sgRNA | gemgroup batch |
| Reference | non-targeting control | control population |
| Target | DESeq2 log2FC with median-of-ratios size factors and shrinkage | DESeq2 log2FC with median-of-ratios size factors |
| Sign check | DESeq2 LFC vs size-factor LFC | DESeq2 LFC vs size-factor LFC |
| Raw-count status | contamination diagnostic only | not load-bearing |

## Increment Test Protocol

| Item | Protocol |
|---|---|
| Baseline | WT gene-identity embedding, optionally plus universal responsiveness for saliency |
| Delta feature | 768-dimensional embedding delta |
| Linear readout | ridge regression |
| Nonlinear readouts | regularization-swept MLP and gradient-boosted trees |
| Validation | 5-fold held-out cross-validation with pooled out-of-fold predictions |
| Evidence excluded | in-sample r |
| Decision criterion | positive held-out increment over the matching baseline |

## Coverage Definition

Tokenization coverage is defined as:

```text
number of sampled input cells whose token sequence contains the target gene
/
number of sampled input cells
```

Cells without the target token are not assigned a zero response and do not contribute to the native-KO estimate. Coverage gates whether the native token-deletion operation was actually applied to enough cells. It does not establish biological specificity; the DESeq2 increment test, universal-responsiveness control, and library-size check remain required.
