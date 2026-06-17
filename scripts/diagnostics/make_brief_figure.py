"""
Headline figure: the embedding delta adds ~0 increment to perturbation-specific effects across every readout.
English labels (avoids matplotlib CJK issues; publication-facing).
All numbers come from the measured output of confound_check / normalize_recheck / path2_specificity.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/figures/figure1_cautionary_case.png"

# -- Panel A: confound baseline vs +delta per target (held-out 5-fold CV r) --
# number sources: raw=confound_check, DESeq2=deseq2_groundtruth (pseudobulk + size factor, primary DESeq2)
settings = ["Raw log2FC\n(contaminated)", "DESeq2 log2FC\n(primary DESeq2)", "DESeq2 saliency\n|log2FC|"]
baseline = [0.693, 0.487, 0.480]   # confound baseline: WT-alone (log2FC) / universal+WT (saliency)
withdelta = [0.690, 0.483, 0.479]  # baseline + embedding delta(768)
incr = [w - b for w, b in zip(withdelta, baseline)]

# -- Panel B: every "apparent positive" collapses to zero under control --
chain_labels = ["oracle dir\n(sign identity)", "thin gate\n[mag,proj]", "rich probe\n[delta]+ctx", "saliency\nrank"]
apparent = [0.39, 0.27, 0.69, -0.34]
controlled = [0.00, 0.01, 0.00, 0.02]   # true values after the matching control (all ~0)
# notes: oracle +0.39 = sign identity -> 0; thin gate [mag,proj] +0.27 is on the contaminated raw target,
#   on the clean DESeq2 target (library-size normalization, sec 2.3) the held-out value is +0.012~0 (measured by thin_gate_clean.py);
#   rich +0.69 -> the WT baseline alone reaches it, delta increment ~0; saliency -0.34 -> partial +0.02 after removing universal responsiveness

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A
x = np.arange(len(settings)); w = 0.36
axA.bar(x - w/2, baseline, w, label="Confound baseline\n(WT / universal responsiveness)", color="#888")
axA.bar(x + w/2, withdelta, w, label="+ embedding delta (768-d)", color="#2c7fb8")
for i in range(len(settings)):
    axA.annotate(f"delta\n= {incr[i]:+.3f}", (x[i] + w/2, withdelta[i] + 0.02),
                 ha="center", fontsize=8.5, color="crimson", fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(settings, fontsize=9)
axA.set_ylabel("Held-out Pearson r (5-fold CV)")
axA.set_ylim(0, 0.92)
axA.set_title("A. Embedding delta adds no measurable increment beyond confounds\n(across every target and readout)", fontsize=11)
axA.legend(fontsize=7.5, loc="upper center", ncol=1)
axA.grid(axis="y", alpha=0.3)

# Panel B
xb = np.arange(len(chain_labels))
axB.bar(xb - w/2, apparent, w, label="Apparent signal", color="#fdae61")
axB.bar(xb + w/2, controlled, w, label="After confound control", color="#2c7fb8")
axB.axhline(0, color="k", lw=0.6)
axB.set_xticks(xb); axB.set_xticklabels(chain_labels, fontsize=8.5)
axB.set_ylabel("Pearson r")
axB.set_title("B. Every apparent positive signal collapses under control\n(controlled estimates approach zero across readouts)", fontsize=11)
axB.legend(fontsize=8, loc="lower right")
axB.grid(axis="y", alpha=0.3)

fig.suptitle("Frangieh: Geneformer in-silico KO embedding delta carries no perturbation-specific signal",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace(".png", ".pdf"))   # vector version (for submission)
print(f"figure saved: {OUT} (300dpi PNG + PDF)")
