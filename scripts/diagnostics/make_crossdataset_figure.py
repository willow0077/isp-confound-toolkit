"""
Cross-dataset summary figure: the delta increment is ~0 across 2 datasets / multiple perturbations / linear+nonlinear.
Numbers from deseq2_groundtruth (Frangieh) / nonlinear_seal / deseq2_replogle (5 perturbations). English labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/figures/figure2_crossdataset.png"

# Panel A: WT-alone vs WT+delta (DESeq2 target, held-out r)
condA = ["Frangieh\nCCND1", "Replogle\nHSPA9", "Replogle\nSUPT6H",
         "Replogle\nCSE1L", "Replogle\nDHX15", "Replogle\nGATA1"]
# the 5 Replogle values use the precise r_wt/r_dw from *_replogle_deseq2.npz (avoids the false precision
# of subtracting rounded bars, e.g. GATA1 0.497-0.494=-0.003 but the true increment is -0.00359->-0.004); CCND1 is Frangieh DESeq2.
wt    = [0.487, 0.5938, 0.5767, 0.3582, 0.5514, 0.4974]
wtd   = [0.483, 0.5778, 0.5732, 0.3441, 0.5432, 0.4938]
incr  = [w2 - w1 for w1, w2 in zip(wt, wtd)]

# Panel B: delta increment -- across readouts (Frangieh linear+nonlinear) + across perturbations (Replogle)
condB = ["Frangieh\nridge", "Frangieh\nMLP", "Frangieh\nGBT",
         "HSPA9", "SUPT6H", "CSE1L", "DHX15", "GATA1"]
incrB = [-0.004, -0.024, +0.010, -0.016, -0.004, -0.014, -0.008, -0.004]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(condA)); w = 0.38
axA.bar(x - w/2, wt, w, label="gene-identity baseline (WT)", color="#888")
axA.bar(x + w/2, wtd, w, label="+ embedding delta (768-d)", color="#2c7fb8")
for i in range(len(condA)):
    axA.annotate(f"{incr[i]:+.3f}", (x[i], max(wt[i], wtd[i]) + 0.015),
                 ha="center", fontsize=8.5, color="crimson", fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(condA, fontsize=8.5)
axA.set_ylabel("Held-out Pearson r (DESeq2 target, 5-fold CV)")
axA.set_ylim(0, 0.72)
axA.set_title("A. Delta adds ~zero beyond baseline\n(2 datasets, 5 cross-process perturbations)", fontsize=11)
axA.legend(fontsize=8, loc="upper right"); axA.grid(axis="y", alpha=0.3)

xb = np.arange(len(condB))
colors = ["#6a3d9a"]*3 + ["#2c7fb8"]*5
axB.bar(xb, incrB, color=colors)
axB.axhspan(-0.03, 0.03, color="gray", alpha=0.15, label="±0.03 'zero signal' band")
axB.axhline(0, color="k", lw=0.6)
axB.set_xticks(xb); axB.set_xticklabels(condB, fontsize=8.5, rotation=20, ha="right")
axB.set_ylabel("Delta increment (held-out r)")
axB.set_ylim(-0.06, 0.06)
axB.set_title("B. Robust across readouts (linear+nonlinear)\nand perturbations — no positive increment", fontsize=11)
axB.legend(fontsize=8, loc="lower right"); axB.grid(axis="y", alpha=0.3)
axB.text(1, -0.045, "Frangieh: ridge/MLP/GBT", fontsize=7.5, color="#6a3d9a", ha="center")
axB.text(5.5, -0.045, "Replogle (DESeq2)", fontsize=7.5, color="#2c7fb8", ha="center")

fig.suptitle("Geneformer in-silico KO embedding delta carries no perturbation-specific signal beyond a gene-identity baseline",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace(".png", ".pdf"))   # vector version (for submission)
print(f"figure saved: {OUT} (300dpi PNG + PDF)")
