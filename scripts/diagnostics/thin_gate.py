"""Thin gate: best linear-readout ceiling from the 2 cached features (mag, proj)."""
import numpy as np
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

d = np.load("benchmark_output/pathd_cache/CCND1_cache.npz", allow_pickle=True)
gene, cellpos, cos, proj = d["gene"], d["cellpos"], d["cos"], d["proj"]
n = int(d["n_valid"])
gt = {int(t): float(f) for t, f in zip(d["gt_tokens"], d["gt_fc"])}
uniq = np.unique(gene); t2c = {t: i for i, t in enumerate(uniq)}
col = np.array([t2c[t] for t in gene])
C = np.full((n, len(uniq)), np.nan, np.float32); P = np.full((n, len(uniq)), np.nan, np.float32)
C[cellpos, col] = cos; P[cellpos, col] = proj
mag = np.nanmean(1 - C, 0); pj = np.nanmean(P, 0)
y = np.array([gt.get(int(t), np.nan) for t in uniq])
m = ~np.isnan(mag) & ~np.isnan(y) & (y != 0)
mag, pj, y = mag[m], pj[m], y[m]

print(f"valid genes {m.sum()}")
print(f"proj alone vs log2fc          : Pearson {pearsonr(pj, y)[0]:+.4f}")
print(f"signed_shift(mag*sign(proj))  : Pearson {pearsonr(mag * np.sign(pj), y)[0]:+.4f}")
X = np.column_stack([mag, pj, np.ones_like(mag)])
beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
print(f"best linear [mag,proj]->log2fc: multiple R {pearsonr(yhat, y)[0]:+.4f}  (2-feature linear ceiling)")
X2 = np.column_stack([mag, pj, np.abs(pj), mag * np.sign(pj), np.ones_like(mag)])
beta2, *_ = np.linalg.lstsq(X2, y, rcond=None); yhat2 = X2 @ beta2
print(f"with |proj| and mag*sign(proj): multiple R {pearsonr(yhat2, y)[0]:+.4f}")
