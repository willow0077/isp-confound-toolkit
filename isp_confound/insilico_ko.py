"""
isp_confound.insilico_ko
========================
Path-D implementation of Geneformer's native in-silico KO operator --
diagnostic infrastructure, NOT a validated perturbation predictor (see manuscript Sec. 2).

==============================================================================
 IMPORTANT NOTICE (consistent with the accompanying paper's conclusion)
------------------------------------------------------------------------------
 The KO embedding shifts / "direction" / "confidence" produced by this module
 are NOT validated perturbation predictions. The paper's central empirical
 finding is precisely that the frozen Geneformer native in-silico KO embedding
 delta carries no perturbation-specific signal beyond a gene-identity baseline.
 Concretely:
   * `mean_shift` is just 1 - cosine -- a confound-dominated "universal
     responsiveness", not a specific effect;
   * `direction` (up/down) is a mechanical rank/projection inference, not
     calibrated against ground truth;
   * `confidence` is only the Monte-Carlo sampling consistency, not predictive
     reliability.
 This module is used only as a delta/shift generator feeding the confound-
 diagnostic scripts. Do not present its outputs as deployable KO-effect
 predictions.
==============================================================================

Wraps Geneformer's native InSilicoPerturber, providing:
  - cell-type-stratified delta generation
  - Monte-Carlo sampling consistency (not predictive confidence)
  - structured output (KOResult)
  - batched multi-gene delta generation

WARNING: import order matters: datasets -> peft -> geneformer, then transformers.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from isp_confound.geneformer_wrapper import _lazy_import

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Structured output
# ------------------------------------------------------------------

@dataclass
class KOResult:
    """
    Structured output of a single in-silico KO delta computation.

    NOTE: not a validated perturbation prediction (see the module-level notice).
    `mean_shift` / `direction` / `confidence` are diagnostic quantities, not
    calibrated perturbation effects.

    Attributes
    ----------
    target_gene : str
        Knocked-out gene name.
    cell_type : str
        Cell type the prediction was run on ("all" means no stratification).
    n_cells : int
        Number of cells used in the prediction.
    de_genes : pd.DataFrame
        Differential-expression table with columns:
        gene, mean_shift, std_shift, direction, confidence.
        mean_shift = 1 - cosine_similarity(KO_emb, WT_emb); larger = more affected.
        direction is derived from the embedding shift (see _get_emb_based_directions).
    mean_shift : np.ndarray
        Mean expression-change vector across all genes (KO minus WT).
    std_shift : np.ndarray
        Uncertainty (std over Monte Carlo Dropout repeats).
    gene_names : list[str]
        Gene names corresponding to mean_shift / std_shift.

    Examples
    --------
    >>> result = ko.predict(adata, "PDCD1", cell_type="CD8_exhausted")
    >>> print(result.top_upregulated(n=10))
    >>> result.to_dataframe().to_csv("ko_result.csv")
    """

    target_gene: str
    cell_type: str
    n_cells: int
    de_genes: pd.DataFrame
    mean_shift: np.ndarray
    std_shift: np.ndarray
    gene_names: List[str]

    def top_upregulated(self, n: int = 20) -> pd.DataFrame:
        """Return the n genes with the largest predicted up-regulation."""
        return (
            self.de_genes[self.de_genes["direction"] == "up"]
            .nlargest(n, "mean_shift")
            .reset_index(drop=True)
        )

    def top_downregulated(self, n: int = 20) -> pd.DataFrame:
        """Return the n most-affected genes whose direction is down-regulation."""
        return (
            self.de_genes[self.de_genes["direction"] == "down"]
            .nlargest(n, "mean_shift")
            .reset_index(drop=True)
        )

    def high_confidence(self, min_confidence: float = 0.7) -> pd.DataFrame:
        """Return predictions with confidence above a threshold."""
        return self.de_genes[
            self.de_genes["confidence"] >= min_confidence
        ].reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full differential-expression table."""
        return self.de_genes.copy()

    def summary(self) -> str:
        """Return a text summary of the prediction."""
        n_up = (self.de_genes["direction"] == "up").sum()
        n_down = (self.de_genes["direction"] == "down").sum()
        n_high_conf = (self.de_genes["confidence"] >= 0.7).sum()
        return (
            f"KO prediction: {self.target_gene} -> {self.cell_type}\n"
            f"  cells used: {self.n_cells}\n"
            f"  up-regulated: {n_up}  down-regulated: {n_down}\n"
            f"  high-confidence (>=0.7): {n_high_conf}"
        )


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class InSilicoKO:
    """
    Cell-type-specific in-silico gene-knockout delta generator
    (diagnostic input, NOT a validated predictor -- see the module-level notice).

    Wraps Geneformer's native InSilicoPerturber, providing:
    - cell-type-stratified delta generation (different states reported separately)
    - Monte Carlo Dropout sampling consistency (not predictive confidence)
    - structured KOResult output

    Parameters
    ----------
    wrapper : GeneformerWrapper
        A loaded GeneformerWrapper instance.
    n_mc_samples : int
        Monte Carlo Dropout repeats; more is a better confidence estimate but slower.
        5 is fine during development, 20-50 before publication.
    cell_intoken_size : int
        Cells processed per perturber call; controls GPU memory.
        10-20 is reasonable on a 4 GB T400.

    Examples
    --------
    >>> from isp_confound import GeneformerWrapper, InSilicoKO
    >>> wrapper = GeneformerWrapper().load()
    >>> ko = InSilicoKO(wrapper)
    >>> result = ko.predict(adata, target_gene="PDCD1", cell_type="CD8_exhausted")
    >>> print(result.summary())
    >>> print(result.top_upregulated(10))
    """

    def __init__(
        self,
        wrapper: "GeneformerWrapper",
        n_mc_samples: int = 10,
        cell_intoken_size: int = 10,
    ):
        self.wrapper = wrapper
        self.n_mc_samples = n_mc_samples
        self.cell_intoken_size = cell_intoken_size

    # ------------------------------------------------------------------
    # Main prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        adata: "anndata.AnnData",
        target_gene: str,
        cell_type: Optional[str] = None,
        cell_type_key: str = "cell_type",
        perturbation_type: str = "delete",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> KOResult:
        """
        Run a single-gene in-silico KO prediction for a given cell type.

        Parameters
        ----------
        adata : AnnData
            Input single-cell data (should already pass validate_adata).
        target_gene : str
            Target gene name (HGNC symbol, e.g. "PDCD1", "TOX").
        cell_type : str, optional
            Target cell type. If None, all cells are used.
            e.g. "CD8_exhausted", "Treg".
        cell_type_key : str
            Cell-type column name, default "cell_type".
        perturbation_type : str
            Perturbation type, "delete" (KO) or "overexpress" (OE).
        output_dir : str | Path, optional
            Directory for intermediate results; a temporary directory if None.

        Returns
        -------
        KOResult
            Structured prediction result.

        Raises
        ------
        ValueError
            The target gene is not in the Geneformer vocabulary.
        ValueError
            The requested cell type does not exist in the data.

        Examples
        --------
        >>> result = ko.predict(adata, "PDCD1", cell_type="CD8_exhausted")
        >>> result.top_upregulated(10)
        """
        self.wrapper._check_loaded()

        # Map gene symbol -> Ensembl ID (Geneformer works in Ensembl IDs internally)
        gene2id = self.wrapper.get_gene2id()
        target_ensembl = self._resolve_gene_id(target_gene, adata, gene2id)

        if target_ensembl not in gene2id:
            raise ValueError(
                f"Gene '{target_gene}' (Ensembl: {target_ensembl}) is not in the Geneformer vocabulary.\n"
                f"Check that the gene name is an HGNC symbol (e.g. PDCD1, TOX, CD8A)."
            )

        # Subset to the target cell type
        adata_subset = self._subset_by_cell_type(
            adata, cell_type, cell_type_key
        )
        ct_label = cell_type if cell_type else "all"
        logger.info(
            f"In-silico KO: {target_gene} -> {ct_label}, "
            f"cells: {adata_subset.n_obs}"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(output_dir) if output_dir else Path(tmp_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: AnnData -> loom format
            loom_path = self.wrapper._adata_to_loom(
                adata_subset, work_dir, cell_type_key
            )

            # Step 2: loom -> tokenized HuggingFace Dataset
            tokenized_dir = work_dir / "tokenized"
            tokenized_dir.mkdir(exist_ok=True)
            self._tokenize_loom(loom_path, tokenized_dir)
            tokenized_dataset_dir = tokenized_dir / "tokenized.dataset"

            # Step 3: run Geneformer's native InSilicoPerturber
            shifts, gene_names, directions = self._run_perturber(
                tokenized_path=str(tokenized_dataset_dir),
                target_gene=target_ensembl,
                perturbation_type=perturbation_type,
                work_dir=work_dir,
            )

        # Build the structured output
        return self._build_result(
            target_gene=target_gene,
            cell_type=ct_label,
            n_cells=adata_subset.n_obs,
            shifts=shifts,
            gene_names=gene_names,
            directions=directions,
        )

    def predict_batch(
        self,
        adata: "anndata.AnnData",
        target_genes: List[str],
        cell_type: Optional[str] = None,
        cell_type_key: str = "cell_type",
        perturbation_type: str = "delete",
    ) -> Dict[str, KOResult]:
        """
        Predict KO effects for several genes.

        Parameters
        ----------
        target_genes : list[str]
            Target gene list.
        Other parameters are as in predict().

        Returns
        -------
        dict[str, KOResult]
            gene_name -> KOResult mapping.

        Examples
        --------
        >>> checkpoint_genes = ["PDCD1", "HAVCR2", "LAG3", "TOX"]
        >>> results = ko.predict_batch(adata, checkpoint_genes, cell_type="CD8_exhausted")
        >>> for gene, result in results.items():
        ...     print(result.summary())
        """
        results = {}
        n = len(target_genes)
        for i, gene in enumerate(target_genes, 1):
            logger.info(f"Batch prediction progress: {i}/{n} -- {gene}")
            try:
                results[gene] = self.predict(
                    adata,
                    target_gene=gene,
                    cell_type=cell_type,
                    cell_type_key=cell_type_key,
                    perturbation_type=perturbation_type,
                )
            except ValueError as e:
                logger.warning(f"Skipping {gene}: {e}")
        return results

    def compare_cell_types(
        self,
        adata: "anndata.AnnData",
        target_gene: str,
        cell_types: List[str],
        cell_type_key: str = "cell_type",
    ) -> Dict[str, KOResult]:
        """
        Compare the same gene's KO effect across cell types.

        The same gene can have different in-silico KO effects in different cell
        states, so the prediction is run per cell type and returned separately.

        Parameters
        ----------
        target_gene : str
            Target gene name.
        cell_types : list[str]
            Cell types to compare,
            e.g. ["CD8_exhausted", "CD8_effector", "Treg"].

        Returns
        -------
        dict[str, KOResult]
            cell_type -> KOResult mapping.

        Examples
        --------
        >>> results = ko.compare_cell_types(
        ...     adata, "PDCD1",
        ...     ["CD8_exhausted", "CD8_effector", "Treg"]
        ... )
        >>> for ct, result in results.items():
        ...     print(f"{ct}: {result.top_upregulated(5)['gene'].tolist()}")
        """
        results = {}
        for ct in cell_types:
            logger.info(f"Comparing cell type: {ct}")
            try:
                results[ct] = self.predict(
                    adata,
                    target_gene=target_gene,
                    cell_type=ct,
                    cell_type_key=cell_type_key,
                )
            except ValueError as e:
                logger.warning(f"Skipping {ct}: {e}")
        return results

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _resolve_gene_id(
        self,
        gene_symbol: str,
        adata: "anndata.AnnData",
        gene2id: dict,
    ) -> str:
        """
        Resolve a gene symbol to the Ensembl ID Geneformer uses.

        Priority:
        1. If gene_symbol is already an Ensembl ID (ENSG...), return it.
        2. Look up gene symbol -> Ensembl ID in adata.var.
        3. Fall back to the original input (it may already be a key in the token dict).
        """
        # Already an Ensembl ID
        if gene_symbol.startswith("ENSG"):
            return gene_symbol

        # Look up the mapping in the data (same column priority as _adata_to_loom)
        if gene_symbol in adata.var_names:
            _EID_CANDIDATES = ("ensembl_id", "ensemble_id", "gene_id", "gene_ids")
            for col in _EID_CANDIDATES:
                if col in adata.var.columns:
                    eid = adata.var.loc[gene_symbol, col]
                    if isinstance(eid, pd.Series):
                        eid = eid.iloc[0]
                    if isinstance(eid, str) and eid.startswith("ENSG"):
                        return str(eid)

        # Fallback: check whether the raw gene_symbol is directly in the token dict
        if gene_symbol in gene2id:
            return gene_symbol

        return gene_symbol

    def _subset_by_cell_type(
        self,
        adata: "anndata.AnnData",
        cell_type: Optional[str],
        cell_type_key: str,
    ) -> "anndata.AnnData":
        """Subset AnnData to a single cell type."""
        if cell_type is None:
            return adata

        if cell_type_key not in adata.obs.columns:
            raise ValueError(
                f"Cell-type column '{cell_type_key}' not found; "
                f"available columns: {list(adata.obs.columns)}"
            )

        mask = adata.obs[cell_type_key] == cell_type
        subset = adata[mask].copy()

        if subset.n_obs == 0:
            available = adata.obs[cell_type_key].unique().tolist()
            raise ValueError(
                f"Cell type '{cell_type}' does not exist in the data.\n"
                f"Available types: {available}"
            )

        return subset

    def _tokenize_loom(
        self,
        loom_path: str,
        output_dir: Path,
    ) -> None:
        """
        Use Geneformer's TranscriptomeTokenizer to convert a loom file into a
        tokenized HuggingFace Dataset (the input format for InSilicoPerturber).
        """
        geneformer, _ = _lazy_import()

        tk = geneformer.TranscriptomeTokenizer(
            custom_attr_name_dict={},
            nproc=1,
            model_version="V2",
        )
        tk.tokenize_data(
            data_directory=str(Path(loom_path).parent),
            output_directory=str(output_dir),
            output_prefix="tokenized",
            file_format="loom",
        )

    def _run_perturber(
        self,
        tokenized_path: str,
        target_gene: str,
        perturbation_type: str,
        work_dir: Path,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Call Geneformer's native InSilicoPerturber and return the perturbation-score
        vector, the gene-name list, and the direction list.

        perturbation score = 1 - cosine_similarity(KO_emb, WT_emb): larger means a
        larger embedding change, i.e. more affected by the perturbation.

        Direction (embedding-based): run a WT and a KO forward pass and project the
        embedding change onto an "expression axis"; falls back to rank-based on failure.

        MC alignment: each MC sample is indexed by gene name and aligned with
        pd.concat(..., join="inner") to avoid order mismatches.
        """
        geneformer, _ = _lazy_import()

        model_path = (
            str(self.wrapper.model_dir)
            if self.wrapper.model_dir
            else "ctheodoris/Geneformer"
        )

        perturb_dir = work_dir / "perturb_output"
        perturb_dir.mkdir(exist_ok=True)

        # InSilicoPerturber loads its own copy of the model from disk (uses GPU memory).
        # Move the wrapper model to CPU first to avoid two copies on the GPU (OOM).
        import torch
        wrapper_model = self.wrapper._model
        model_device = next(wrapper_model.parameters()).device
        if model_device.type != "cpu":
            wrapper_model.cpu()
            torch.cuda.empty_cache()
            logger.info("wrapper model temporarily moved to CPU (freeing GPU for InSilicoPerturber)")

        # Run repeatedly (Monte Carlo Dropout), aligned by gene
        all_samples: List[pd.Series] = []
        gene_to_token_id: Dict[str, int] = {}  # collected from the first valid sample

        try:
            for i in range(self.n_mc_samples):
                sample_dir = perturb_dir / f"sample_{i}"
                sample_dir.mkdir(exist_ok=True)

                isp = geneformer.InSilicoPerturber(
                    perturb_type=perturbation_type,
                    perturb_rank_shift=None,
                    genes_to_perturb=[target_gene],
                    combos=0,
                    anchor_gene=None,
                    model_type="Pretrained",
                    num_classes=0,
                    emb_mode="cls_and_gene",
                    cell_emb_style="mean_pool",
                    filter_data=None,
                    cell_states_to_model=None,
                    max_ncells=None,
                    emb_layer=-1,
                    forward_batch_size=self.cell_intoken_size,
                    nproc=1,
                    model_version="V2",
                )

                isp.perturb_data(
                    model_directory=model_path,
                    input_data_file=tokenized_path,
                    output_directory=str(sample_dir),
                    output_prefix=f"ko_{i}",
                )
                # Explicitly free the perturber's model weights to avoid accumulating GPU memory across samples
                del isp
                torch.cuda.empty_cache()

                # Read this sample's result (returns a 3-tuple)
                shifts_i, gene_names_i, token_ids_i = self._parse_perturber_output(sample_dir)

                if len(shifts_i) == 0:
                    logger.warning(f"MC sample {i} returned an empty result, skipping")
                    continue

                # Build the gene -> token_id map from the first valid sample
                if not gene_to_token_id:
                    gene_to_token_id = dict(zip(gene_names_i, token_ids_i))

                # Index by gene name for later alignment (shifts_i[0] is the perturbation_score)
                all_samples.append(
                    pd.Series(shifts_i[0], index=gene_names_i, name=f"sample_{i}")
                )
        finally:
            # Whether success or failure, move the wrapper model back to its device
            wrapper_model.to(model_device)
            torch.cuda.empty_cache()
            logger.info(f"wrapper model moved back to {model_device}")

        if not all_samples:
            logger.warning("All MC samples were empty")
            return np.zeros((2, 0)), [], []

        # Inner join: keep only genes present in every sample
        aligned = pd.concat(all_samples, axis=1, join="inner")  # [n_genes, n_samples]
        gene_names = aligned.index.tolist()
        mean_shifts = aligned.mean(axis=1).values.astype(np.float32)
        std_shifts = aligned.std(axis=1).fillna(0).values.astype(np.float32)

        # Rebuild token_ids in the aligned gene-name order
        token_ids = [gene_to_token_id.get(g, -1) for g in gene_names]

        # Determine direction from the embedding change (falls back to rank-based on failure)
        gene2id = self.wrapper.get_gene2id()
        target_token_id = gene2id.get(target_gene, -1)
        directions = self._get_emb_based_directions(tokenized_path, target_token_id, token_ids)

        return np.stack([mean_shifts, std_shifts], axis=0), gene_names, directions

    def _get_emb_based_directions(
        self,
        tokenized_path: str,
        target_token_id: int,
        token_ids: List[int],
    ) -> List[str]:
        """
        Determine perturbation direction (up/down) by projecting the embedding change
        onto an "expression axis".

        Idea:
        1. expression axis = mean(high-expression gene embeddings) - mean(low-expression gene embeddings)
           high = genes at positions 1..N_BOUNDARY in the tokenized sequence
           low  = the last N_BOUNDARY genes of the sequence (skipping EOS)
        2. For each affected gene g, average WT/KO embeddings across cells:
           delta = mean_KO_emb(g) - mean_WT_emb(g)
           direction = "up" if dot(delta, expr_axis) > 0 else "down"

        Any failure (OOM, dataset load error, ...) falls back to rank-based.
        """
        import torch

        try:
            from datasets import load_from_disk
            ds = load_from_disk(tokenized_path)
        except Exception as e:
            logger.warning(f"embedding-based direction: cannot load dataset, falling back to rank-based: {e}")
            return self._get_rank_directions(tokenized_path, target_token_id, token_ids)

        if target_token_id < 0:
            return self._get_rank_directions(tokenized_path, target_token_id, token_ids)

        model = self.wrapper._model
        device = self.wrapper.device

        try:
            hidden_size = model.config.hidden_size
        except AttributeError:
            hidden_size = 512

        N_BOUNDARY = 50  # number of high/low-expression genes used to define the expression axis
        token_id_set = set(t for t in token_ids if t >= 0)

        # Accumulate WT/KO embeddings across cells (stored on CPU to save GPU memory)
        wt_emb_sum: Dict[int, np.ndarray] = {}
        ko_emb_sum: Dict[int, np.ndarray] = {}
        emb_count: Dict[int, int] = {}
        expr_axis_sum = np.zeros(hidden_size, dtype=np.float64)
        expr_axis_count = 0

        try:
            with torch.no_grad():
                for cell_ids in ds["input_ids"]:
                    cell_ids_list = list(cell_ids)
                    if target_token_id not in cell_ids_list:
                        continue
                    seq_len = len(cell_ids_list)
                    if seq_len < 2 * N_BOUNDARY + 2:
                        continue

                    # WT forward pass
                    wt_t = torch.tensor([cell_ids_list], dtype=torch.long, device=device)
                    wt_mask = torch.ones_like(wt_t)
                    wt_out = model(input_ids=wt_t, attention_mask=wt_mask)
                    wt_h = wt_out.last_hidden_state[0].float().cpu().numpy()  # [seq_len, H]

                    # Expression axis: high-expression (positions 1..N) vs low-expression (last N before EOS)
                    top_mean = wt_h[1:N_BOUNDARY + 1].mean(axis=0)
                    bot_mean = wt_h[-N_BOUNDARY - 1:-1].mean(axis=0)
                    expr_axis_sum += (top_mean - bot_mean).astype(np.float64)
                    expr_axis_count += 1

                    # KO forward pass: delete the target token
                    ko_ids = [t for t in cell_ids_list if t != target_token_id]
                    ko_t = torch.tensor([ko_ids], dtype=torch.long, device=device)
                    ko_mask = torch.ones_like(ko_t)
                    ko_out = model(input_ids=ko_t, attention_mask=ko_mask)
                    ko_h = ko_out.last_hidden_state[0].float().cpu().numpy()  # [seq_len-1, H]

                    wt_pos = {tid: pos for pos, tid in enumerate(cell_ids_list)}
                    ko_pos = {tid: pos for pos, tid in enumerate(ko_ids)}

                    for tid in token_id_set:
                        if tid in wt_pos and tid in ko_pos:
                            p_wt, p_ko = wt_pos[tid], ko_pos[tid]
                            if tid not in wt_emb_sum:
                                wt_emb_sum[tid] = np.zeros(hidden_size, dtype=np.float64)
                                ko_emb_sum[tid] = np.zeros(hidden_size, dtype=np.float64)
                                emb_count[tid] = 0
                            wt_emb_sum[tid] += wt_h[p_wt].astype(np.float64)
                            ko_emb_sum[tid] += ko_h[p_ko].astype(np.float64)
                            emb_count[tid] += 1

        except Exception as e:
            logger.warning(f"embedding-based direction failed ({e}), falling back to rank-based")
            return self._get_rank_directions(tokenized_path, target_token_id, token_ids)

        if expr_axis_count == 0:
            logger.warning("Could not build the expression axis (no valid cells), falling back to rank-based")
            return self._get_rank_directions(tokenized_path, target_token_id, token_ids)

        expr_axis = expr_axis_sum / expr_axis_count
        norm = np.linalg.norm(expr_axis)
        if norm < 1e-8:
            logger.warning("Expression axis is near-zero, falling back to rank-based")
            return self._get_rank_directions(tokenized_path, target_token_id, token_ids)
        expr_axis = expr_axis / norm

        directions = []
        for tid in token_ids:
            if tid < 0 or tid not in emb_count or emb_count[tid] == 0:
                directions.append("unknown")
                continue
            delta = (ko_emb_sum[tid] - wt_emb_sum[tid]) / emb_count[tid]
            proj = float(np.dot(delta, expr_axis))
            directions.append("up" if proj > 0 else "down")

        n_up = directions.count("up")
        n_down = directions.count("down")
        n_unk = directions.count("unknown")
        logger.info(f"Embedding-based direction counts: up={n_up}, down={n_down}, unknown={n_unk}")
        return directions

    def _get_rank_directions(
        self,
        tokenized_path: str,
        target_token_id: int,
        token_ids: List[int],
    ) -> List[str]:
        """
        Determine perturbation direction from gene position (rank) in the tokenized sequence.

        Geneformer orders gene tokens by descending expression, so a smaller position
        means higher expression. After deleting the target gene, genes positioned after
        it (position > target_position) rise one rank and are treated as relatively
        up-regulated ("up"); otherwise "down".
        """
        try:
            from datasets import load_from_disk
            ds = load_from_disk(tokenized_path)
        except Exception as e:
            logger.warning(f"Cannot load tokenized dataset; setting directions to 'unknown': {e}")
            return ["unknown"] * len(token_ids)

        if target_token_id < 0:
            logger.warning("Invalid target gene token ID; setting directions to 'unknown'")
            return ["unknown"] * len(token_ids)

        all_ids = ds["input_ids"]  # list of lists

        # Build a {token_id: position} map per cell (O(1) lookup)
        cell_pos_maps = [{tid: pos for pos, tid in enumerate(ids)} for ids in all_ids]

        # Mean position of the target gene
        target_positions = [m[target_token_id] for m in cell_pos_maps if target_token_id in m]
        if not target_positions:
            logger.warning(f"Target gene token {target_token_id} appears in no cell")
            return ["unknown"] * len(token_ids)

        target_mean_pos = float(np.mean(target_positions))
        logger.info(
            f"Target gene token {target_token_id} mean rank position: "
            f"{target_mean_pos:.1f} (smaller = higher expression)"
        )

        directions = []
        for tid in token_ids:
            if tid < 0:
                directions.append("unknown")
                continue
            gene_positions = [m[tid] for m in cell_pos_maps if tid in m]
            if not gene_positions:
                directions.append("unknown")
            else:
                gene_mean_pos = float(np.mean(gene_positions))
                # position > target position -> gene is lower-expressed than target -> relatively up after KO
                directions.append("up" if gene_mean_pos > target_mean_pos else "down")

        n_up = directions.count("up")
        n_down = directions.count("down")
        logger.info(f"Rank-based direction counts: up={n_up}, down={n_down}, unknown={len(directions)-n_up-n_down}")
        return directions

    def _parse_perturber_output(
        self, output_dir: Path
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Parse InSilicoPerturber output files; return perturbation scores, gene names, and token IDs.

        aggregate_gene_shifts output:
        - Cosine_sim_mean = cosine_similarity(KO_emb, WT_emb) per affected gene
        - closer to 1 = smaller embedding change = less affected
        - converted to a perturbation score: perturbation_score = 1 - Cosine_sim_mean (larger = more affected)

        Returns:
        - shifts[0] = perturbation_score (1 - cosine_sim_mean)
        - shifts[1] = cosine_sim_stdev (the std needs no conversion)
        - gene_names: affected gene names
        - token_ids: corresponding token IDs (used for rank-based direction)
        """
        geneformer, _ = _lazy_import()

        stats = geneformer.InSilicoPerturberStats(
            mode="aggregate_gene_shifts",
            genes_perturbed="all",
            combos=0,
            anchor_gene=None,
            cell_states_to_model=None,
            model_version="V2",
        )

        # Generate the stats file
        stats.get_stats(
            str(output_dir),
            str(output_dir),
            str(output_dir),
            "stats",
        )

        # Read the stats result
        stats_files = list(output_dir.glob("*stats*.csv"))
        if not stats_files:
            stats_files = list(output_dir.glob("*stats*.pkl"))

        if not stats_files:
            logger.warning(f"No stats output file found; directory contents: {list(output_dir.iterdir())}")
            return np.array([]), [], []

        stats_file = stats_files[0]
        if stats_file.suffix == ".pkl":
            df = pd.read_pickle(stats_file)
        else:
            df = pd.read_csv(stats_file)
        # Drop rows where Affected_gene_name is NaN (the cell_emb row, i.e. the KO gene on itself)
        df = df.dropna(subset=["Affected_gene_name"])

        gene_names = df["Affected_gene_name"].tolist()
        token_ids = df["Affected"].astype(int).tolist()

        # perturbation score = 1 - cosine_sim (larger = larger embedding change = more affected)
        perturbation_scores = (1.0 - df["Cosine_sim_mean"].values).astype(np.float32)
        std_shifts = df["Cosine_sim_stdev"].values.astype(np.float32)

        return np.stack([perturbation_scores, std_shifts], axis=0), gene_names, token_ids

    def _build_result(
        self,
        target_gene: str,
        cell_type: str,
        n_cells: int,
        shifts: np.ndarray,
        gene_names: List[str],
        directions: Optional[List[str]] = None,
    ) -> KOResult:
        """
        Convert the raw prediction arrays into a structured KOResult.

        mean_shift = perturbation_score = 1 - cosine_similarity(KO_emb, WT_emb)
        Larger = more affected; results are sorted by this in descending order.

        confidence = 1 - std / (mean + eps); smaller std -> higher agreement across
        MC samples -> higher confidence.

        directions: provided by the direction methods; defaults to all "up" (placeholder) if absent.
        """
        if len(gene_names) == 0:
            logger.warning("Prediction result is empty; returning an empty KOResult")
            empty_df = pd.DataFrame(
                columns=["gene", "mean_shift", "std_shift", "direction", "confidence"]
            )
            return KOResult(
                target_gene=target_gene,
                cell_type=cell_type,
                n_cells=n_cells,
                de_genes=empty_df,
                mean_shift=np.array([]),
                std_shift=np.array([]),
                gene_names=[],
            )

        mean_shifts = shifts[0]
        std_shifts = shifts[1]

        # Confidence: agreement across MC samples (smaller std -> more confident)
        eps = 1e-6
        confidence = 1.0 - (std_shifts / (mean_shifts + eps))
        confidence = np.clip(confidence, 0.0, 1.0)

        # Direction: prefer the computed directions, otherwise a placeholder
        if directions is not None and len(directions) == len(gene_names):
            directions_arr = np.array(directions)
        else:
            directions_arr = np.full(len(gene_names), "unknown")

        de_genes = pd.DataFrame({
            "gene": gene_names,
            "mean_shift": mean_shifts,
            "std_shift": std_shifts,
            "direction": directions_arr,
            "confidence": confidence,
        # Descending by perturbation score: most-affected genes first
        }).sort_values("mean_shift", ascending=False).reset_index(drop=True)

        return KOResult(
            target_gene=target_gene,
            cell_type=cell_type,
            n_cells=n_cells,
            de_genes=de_genes,
            mean_shift=mean_shifts,
            std_shift=std_shifts,
            gene_names=gene_names,
        )
