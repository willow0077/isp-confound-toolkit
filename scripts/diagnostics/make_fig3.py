"""
Figure 3 (path-D's distinct contribution): coverage gating + controlled coverage->estimate analysis.
(A) distribution of real tokenization coverage across Frangieh KO genes -> gating (JAK2/CD274 etc. too low to evaluate).
(B) path-D's machine-precision self-computation makes controlled subsampling feasible: the CCND1 estimate varies
    monotonically with the number of contributing cells, and an ablation localizes it to the direction component.
Data: token_coverage.csv / CCND1_curve_main.csv / CCND1_curve_abl.csv. English labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT = "results/figures/figure3_coverage_pathD.png"

cov = pd.read_csv("results/data/token_coverage.csv")
main = pd.read_csv("results/data/CCND1_curve_main.csv")
abl = pd.read_csv("results/data/CCND1_curve_abl.csv")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: coverage distribution
axA.hist(cov["token_coverage"], bins=25, color="#2c7fb8", alpha=0.85, edgecolor="white")
med = cov["token_coverage"].median()
axA.axvline(med, color="k", ls="--", lw=1, label=f"median {med:.2f}")
for g, c, lab in [("JAK2", 0.023, "2.3%"), ("CD274", 0.107, "11%")]:
    axA.axvline(c, color="crimson", ls=":", lw=1.2)
    axA.annotate(f"{g}\n{lab}", (c, axA.get_ylim()[1]*0.82), color="crimson",
                 fontsize=8.5, ha="center", fontweight="bold")
axA.set_xlabel("Real tokenization coverage (fraction of cells with gene in top-4096)")
axA.set_ylabel("Number of perturbation targets")
axA.set_title("A. Coverage gates which targets are evaluable\n(Frangieh, 217 KO genes)", fontsize=11)
axA.legend(fontsize=8); axA.grid(axis="y", alpha=0.3)

# Panel B: controlled coverage subsampling curve
axB.plot(main["level"], main["r_mean"], "-o", color="#2c7fb8", label="direction resampled (faithful)")
axB.fill_between(main["level"], main["ci_lo"], main["ci_hi"], color="#2c7fb8", alpha=0.2)
axB.plot(abl["level"], abl["r_mean"], "--s", color="#888", markersize=4, label="direction fixed (ablation)")
axB.axhline(0, color="k", lw=0.5)
axB.set_xscale("log")
axB.set_xlabel("Contributing cells (coverage), log scale")
axB.set_ylabel("Recovered Pearson r (CCND1)")
axB.set_title("B. path-D enables controlled coverage-to-estimate analysis\n(machine-precision self-computed cosine, r=1.0 vs official)", fontsize=11)
axB.legend(fontsize=8, loc="lower left"); axB.grid(alpha=0.3)

fig.suptitle("Tokenization coverage gates applicability, and path-D enables controlled coverage analysis",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace(".png", ".pdf"))   # vector version (for submission)
print(f"figure saved: {OUT} (300dpi PNG + PDF)")
