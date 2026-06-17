"""
Generate Supplementary Table S1 (de-circularization matrix) from the authoritative
run log of scripts/pipeline/state_shift_matrix.py.

REQUIRES THE ZENODO CACHE: this generator parses the run log
`benchmark_output/verify_logs/state_matrix_rerun.log`, which is deposited on Zenodo
(see DATA.md), not committed to this repository. It therefore does NOT run from a clean
checkout alone. The committed `Supplementary_Table_S1_decircularization_matrix.md` and
`Supplementary_Table_S1_decircularization_cells.csv` are the static, already-generated
outputs; this script documents how they were produced and re-validates the aggregates.

The state-shift matrix itself needs a GPU + the Geneformer weights, so this generator does
not re-run it; instead it parses the recorded run log (the source of truth), classifies
every cell of the cross-gene projection matrix, writes a machine-readable per-cell CSV,
and re-derives the reported aggregates as a self-consistency check.

Run from the project root: python scripts/diagnostics/make_supp_table_s1.py
"""
import csv
import re
import statistics
from pathlib import Path

LOG = Path("benchmark_output/verify_logs/state_matrix_rerun.log")
OUT_CSV = Path("Supplementary_Table_S1_decircularization_cells.csv")

# Real genes (knockout rows and projection axes) and the mitochondrial cluster, mirroring
# scripts/pipeline/state_shift_matrix.py (GENES, MITO).
REAL_GENES = ["HSPA9", "PHB", "PHB2", "GATA1", "CSE1L", "SUPT6H"]
MITO = {"HSPA9", "PHB", "PHB2"}
PATHWAY = {
    "HSPA9": "mitochondrial (HSP70 chaperone)",
    "PHB": "mitochondrial (prohibitin)",
    "PHB2": "mitochondrial (prohibitin)",
    "GATA1": "erythroid / lineage transcription factor",
    "CSE1L": "nuclear transport",
    "SUPT6H": "transcription elongation",
}


def group_of(gene, is_random):
    if is_random:
        return "random null"
    return "mitochondrial" if gene in MITO else "non-mitochondrial real"


def parse_log():
    """Parse the projection-matrix block: axis order + one row per knockout gene."""
    text = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    axes, rows = None, []
    row_re = re.compile(r"\]\s*([A-Z0-9]+)\s+(-?\d+\.\d+(?:\s+-?\d+\.\d+){5})\s+(\d+)\s*\[(\w+)\]")
    for line in text:
        # header line has all six gene names but no decimal projection values
        if all(g in line for g in REAL_GENES) and not re.search(r"-?\d+\.\d", line):
            # "...] pert\axis  HSPA9  PHB  PHB2  GATA1  CSE1L  SUPT6H   n"
            axes = [t for t in re.split(r"\s+", line.split("]")[-1].strip()) if t in REAL_GENES]
        m = row_re.search(line)
        if m:
            gene = m.group(1)
            vals = [float(x) for x in m.group(2).split()]
            n = int(m.group(3))
            tag = m.group(4)  # mito / other / rand
            rows.append((gene, vals, n, tag))
    if axes != REAL_GENES:
        raise SystemExit(f"axis order from log {axes} != expected {REAL_GENES}")
    if len(rows) != 12:
        raise SystemExit(f"expected 12 knockout rows, parsed {len(rows)}")
    return axes, rows


def classify(a, b, a_is_random):
    """Return (relation, included_in_specificity_comparison)."""
    if a == b:
        return "self / diagonal", False               # self-deletion / circular; not evidence
    if a_is_random:
        return ("null (random→mitochondrial axis)" if b in MITO
                else "null (random→other axis)"), False
    if a in MITO and b in MITO:
        return "same-pathway off-diagonal", True
    if a in MITO and b not in MITO:
        return "cross-pathway off-diagonal", True
    return "real off-diagonal (non-mitochondrial A)", False  # not part of the mito-anchored comparison


def main():
    axes, rows = parse_log()
    random_order = [g for (g, _, _, tag) in rows if tag == "rand"]

    cells = []
    for gene, vals, n, tag in rows:
        is_rand = tag == "rand"
        for b, v in zip(axes, vals):
            relation, included = classify(gene, b, is_rand)
            cells.append({
                "knockout_gene_A": gene,
                "projection_axis_B": b,
                "A_group": group_of(gene, is_rand),
                "B_group": group_of(b, False),
                "relation": relation,
                "projection_value_x1e-3": v,
                "n_contributing_cells": n,
                "included_in_specificity_comparison": included,
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print(f"wrote {OUT_CSV} ({len(cells)} cells)")

    # ── Self-consistency: re-derive the reported aggregates ─────────────
    def mean(sel):
        return statistics.mean(c["projection_value_x1e-3"] for c in cells if sel(c))

    diag = mean(lambda c: c["relation"] == "self / diagonal")
    same = mean(lambda c: c["relation"] == "same-pathway off-diagonal")
    cross = mean(lambda c: c["relation"] == "cross-pathway off-diagonal")
    null_vals = [c["projection_value_x1e-3"] for c in cells
                 if c["relation"] == "null (random→mitochondrial axis)"]
    null_mean = statistics.mean(null_vals)
    null_sd = statistics.pstdev(null_vals)
    z = (same - null_mean) / null_sd

    print(f"random-null genes (draw order, N_RANDOM=6, seed=0): {random_order}")
    print("re-derived aggregates (expected from log in parentheses):")
    print(f"  diagonal mean              = {diag:+.1f} ×1e-3   (+12.4)")
    print(f"  same-pathway off-diagonal  = {same:+.1f} ×1e-3   (+3.7)")
    print(f"  cross-pathway off-diagonal = {cross:+.1f} ×1e-3   (+16.7)")
    print(f"  null (random→mito)         = {null_mean:+.1f} ± {null_sd:.1f} ×1e-3   (-10.0 ± 9.0)")
    print(f"  same-pathway vs null z     = {z:+.2f}   (+1.51)")


if __name__ == "__main__":
    main()
