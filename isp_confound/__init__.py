"""
isp_confound — supporting utilities for the ISP confound-diagnostic toolkit.

These are *supporting utilities*: a thin Geneformer-access layer plus an
in-silico knockout engine, used by the toolkit scripts under ``scripts/``.

The toolkit itself — the reusable, confound-aware evaluation methods (path-D,
universal-responsiveness control, tokenization-coverage gate, library-size
contamination check, de-circularization matrix, heterogeneity pre-check) —
lives in ``scripts/`` (``pipeline/`` and ``diagnostics/``).
"""

from .geneformer_wrapper import GeneformerWrapper, get_geneformer_token_dict_path, load_geneformer_gene2id
from .insilico_ko import InSilicoKO, KOResult

__all__ = ["GeneformerWrapper", "InSilicoKO", "KOResult", "get_geneformer_token_dict_path", "load_geneformer_gene2id"]
