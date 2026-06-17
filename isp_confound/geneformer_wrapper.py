"""
isp_confound.geneformer_wrapper
====================================
Loading Geneformer, tokenization, and embedding extraction.

WARNING: import order matters: datasets -> peft -> geneformer, and only then
   transformers. Violating the order can segfault. The order is managed
   centrally in _lazy_import().
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Geneformer model HuggingFace Hub ID
GENEFORMER_MODEL_ID = "ctheodoris/Geneformer"

# Default token-sequence length (matches pretraining)
DEFAULT_MAX_SEQ_LEN = 4096  # V2 model supports longer sequences

# Geneformer's bundled token dictionary (inside the installed geneformer package)
_GENEFORMER_TOKEN_DICT = "geneformer/token_dictionary_gc104M.pkl"
_GENEFORMER_TOKEN_DICT_FALLBACK = "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"


def _lazy_import():
    """
    Import Geneformer's dependencies in the required order.
    datasets -> peft -> geneformer, and only then transformers.
    Done lazily to avoid module-level import-order problems.
    """
    import datasets          # noqa: F401  must be first
    import peft              # noqa: F401  must be second
    import geneformer        # noqa: F401  third
    import transformers      # noqa: F401  last

    return geneformer, transformers


def _find_token_dict() -> Path:
    """Locate the Geneformer token-dictionary file automatically."""
    gf, _ = _lazy_import()

    pkg_dir = Path(gf.__file__).parent

    # V2 first (matches ctheodoris/Geneformer on HuggingFace, vocab=20275)
    candidates = [
        pkg_dir / "token_dictionary_gc104M.pkl",
        pkg_dir / "gene_dictionaries_30m" / "token_dictionary_gc30M.pkl",
        pkg_dir / "gene_dictionaries" / "token_dictionary_gc30M.pkl",
        pkg_dir / "token_dictionary.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Geneformer token dictionary not found; searched: {[str(c) for c in candidates]}"
    )


def get_geneformer_token_dict_path() -> Path:
    """Return the Geneformer token-dictionary path discovered from the installed package."""
    return _find_token_dict()


def load_geneformer_gene2id(token_dict_path: Optional[Union[str, Path]] = None) -> dict:
    """Load Geneformer's gene-id -> token-id mapping from a discovered or explicit token dictionary."""
    return _load_gene2id_from_token_dict(token_dict_path)


def _load_gene2id_from_token_dict(token_dict_path: Optional[Union[str, Path]] = None) -> dict:
    """
    Load the gene -> token-id mapping from Geneformer's token-dictionary pickle.
    Returns an {ensembl_id: token_id} dict.

    If no path is given, the token dictionary inside the geneformer package is
    located automatically.
    """
    if token_dict_path is None:
        token_dict_path = _find_token_dict()

    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)

    # Drop special tokens (<pad>, <mask>, <cls>, ...)
    return {k: v for k, v in token_dict.items() if not k.startswith("<")}


class GeneformerWrapper:
    """
    Thin wrapper around the pretrained Geneformer model.

    Responsibilities:
    1. Load the pretrained model and the token vocabulary.
    2. Convert AnnData into Geneformer token sequences (via a loom file).
    3. Extract cell embeddings (used by the downstream in-silico KO comparison).

    Parameters
    ----------
    model_dir : str | Path, optional
        Path to a local model directory. If None, downloads from the HuggingFace Hub.
    device : str, optional
        Inference device, "cuda" or "cpu". Auto-detected if None.
    max_seq_len : int
        Maximum gene-sequence length, default 4096 (matches Geneformer V2 pretraining).
    token_dict_path : str | Path, optional
        Path to the Geneformer token-dictionary pickle. Located inside the
        geneformer package automatically if None.

    Examples
    --------
    >>> wrapper = GeneformerWrapper()
    >>> wrapper.load()
    >>> embeddings = wrapper.get_embeddings(adata, cell_type_key="cell_type")
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        token_dict_path: Optional[Union[str, Path]] = None,
    ):
        self.model_dir = Path(model_dir) if model_dir else None
        self.max_seq_len = max_seq_len
        self.token_dict_path = token_dict_path
        self._model = None
        self._gene2id: Optional[Dict[str, int]] = None

        # Auto-detect device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"GeneformerWrapper initialized, device: {self.device}")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load(self) -> "GeneformerWrapper":
        """
        Load the pretrained Geneformer model and token vocabulary.

        If model_dir is None, the model is downloaded from the HuggingFace Hub.
        The token vocabulary is loaded from Geneformer's bundled pickle (not a
        HuggingFace tokenizer).

        Returns
        -------
        self (supports method chaining)
        """
        geneformer, transformers = _lazy_import()

        model_path = str(self.model_dir) if self.model_dir else GENEFORMER_MODEL_ID
        logger.info(f"Loading Geneformer model: {model_path}")

        self._model = transformers.AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()

        # Geneformer uses a custom token dictionary, not a HuggingFace AutoTokenizer
        self._gene2id = _load_gene2id_from_token_dict(self.token_dict_path)
        logger.info(f"Gene vocabulary size: {len(self._gene2id)}")

        return self

    def get_gene2id(self) -> Dict[str, int]:
        """
        Return the gene-id -> Geneformer token-id mapping.
        Used by the in-silico KO engine to resolve target/affected genes to tokens.

        Returns
        -------
        dict
        """
        self._check_loaded()
        return self._gene2id

    def get_embeddings(
        self,
        adata: "anndata.AnnData",
        cell_type_key: Optional[str] = "cell_type",
        batch_size: int = 32,
        output_dir: Optional[Union[str, Path]] = None,
        emb_layer: int = -1,
    ) -> Dict[str, np.ndarray]:
        """
        Extract Geneformer embeddings for every cell in an AnnData.

        Uses Geneformer's native EmbExtractor, returning each cell's transformer
        hidden representation (last layer, mean-pooled, by default).

        Parameters
        ----------
        adata : AnnData
            Input single-cell data.
        cell_type_key : str, optional
            Cell-type column name, used for stratified output.
        batch_size : int
            Inference batch size; 32 is reasonable for a 4 GB T400 GPU.
        output_dir : str | Path, optional
            Directory to save embeddings; a temporary directory is used if None.
        emb_layer : int
            Which layer to take the embedding from; -1 means the last layer.

        Returns
        -------
        dict
            {"embeddings": np.ndarray [n_cells, hidden_dim],
             "cell_types": list[str],
             "cell_ids": list[str]}
        """
        self._check_loaded()
        geneformer, _ = _lazy_import()

        # Convert AnnData to the loom format Geneformer needs (intermediate file)
        with tempfile.TemporaryDirectory() as tmp_dir:
            loom_path = self._adata_to_loom(adata, tmp_dir, cell_type_key)

            out_dir = Path(output_dir) if output_dir else Path(tmp_dir) / "embeddings"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Use Geneformer's native EmbExtractor
            extractor = geneformer.EmbExtractor(
                model_type="Pretrained",
                num_classes=0,
                emb_layer=emb_layer,
                emb_label=["cell_type"] if cell_type_key else [],
                max_ncells=None,
                forward_batch_size=batch_size,
                nproc=1,
            )

            embs = extractor.extract_embs(
                model_directory=str(self.model_dir) if self.model_dir else GENEFORMER_MODEL_ID,
                input_data_file=loom_path,
                output_directory=str(out_dir),
                output_prefix="ispc_emb",
            )

        # Assemble output
        cell_types = []
        if cell_type_key and cell_type_key in adata.obs.columns:
            cell_types = adata.obs[cell_type_key].tolist()

        return {
            "embeddings": np.array(embs),
            "cell_types": cell_types,
            "cell_ids": adata.obs_names.tolist(),
        }

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _check_loaded(self):
        """Ensure the model is loaded, otherwise raise a clear error."""
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded yet; call wrapper.load() first"
            )

    def _adata_to_loom(
        self,
        adata: "anndata.AnnData",
        output_dir: Union[str, Path],
        cell_type_key: Optional[str],
    ) -> str:
        """
        Convert AnnData into the loom format Geneformer requires.

        Geneformer's TranscriptomeTokenizer reads a loom file that contains:
        - the gene expression matrix (raw counts),
        - a column attribute "ensembl_id" (or gene symbol),
        - a row attribute "n_counts" (total UMIs per cell).

        Parameters
        ----------
        adata : AnnData
        output_dir : str | Path
        cell_type_key : str, optional

        Returns
        -------
        str
            Path to the loom file.
        """
        try:
            import loompy
        except ImportError:
            raise ImportError(
                "Please install loompy: pip install loompy\n"
                "loompy is a dependency of the Geneformer tokenizer"
            )
        import scipy.sparse as sp

        output_dir = Path(output_dir)
        loom_path = str(output_dir / "input.loom")

        # Get raw counts
        X = adata.raw.X if adata.raw is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        # Build the row/column attributes loom needs.
        # Geneformer requires the row attribute "ensembl_id" to be Ensembl IDs,
        # not gene symbols. Search by priority: correct spelling first, then
        # tolerate the common misspelling and other formats.
        _EID_CANDIDATES = ("ensembl_id", "ensemble_id", "gene_id", "gene_ids")
        eid_col = next(
            (c for c in _EID_CANDIDATES if c in adata.var.columns),
            None,
        )
        if eid_col:
            ensembl_ids = adata.var[eid_col].tolist()
            # Check coverage: how many genes have a valid ENSG ID
            n_valid = sum(1 for e in ensembl_ids if isinstance(e, str) and e.startswith("ENSG"))
            logger.info(
                f"Ensembl ID source column: '{eid_col}', "
                f"valid ENSG IDs: {n_valid}/{len(ensembl_ids)}"
            )
            if n_valid < len(ensembl_ids) * 0.5:
                logger.warning(
                    f"Over 50% of genes lack a valid Ensembl ID (ENSG...); "
                    f"tokenization coverage may be very low. "
                    f"Check the dataset's var['{eid_col}'] column."
                )
        else:
            ensembl_ids = adata.var_names.tolist()
            logger.warning(
                f"No Ensembl ID column found (tried: {_EID_CANDIDATES}); "
                f"falling back to var_names ({adata.var_names[:3].tolist()}...). "
                f"If var_names are HGNC symbols, tokenization coverage will be near 0."
            )
        row_attrs = {
            "ensembl_id": ensembl_ids,
        }

        n_counts = X.sum(axis=1)
        col_attrs = {
            "CellID": adata.obs_names.tolist(),
            "n_counts": n_counts.tolist(),
        }
        if cell_type_key and cell_type_key in adata.obs.columns:
            col_attrs["cell_type"] = adata.obs[cell_type_key].tolist()

        # loompy expects genes x cells (transpose)
        loompy.create(loom_path, X.T, row_attrs, col_attrs)
        logger.info(f"loom file created: {loom_path} ({adata.n_obs} cells x {adata.n_vars} genes)")

        return loom_path
