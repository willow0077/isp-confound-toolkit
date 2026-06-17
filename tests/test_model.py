"""
tests/test_model.py
===================
Unit tests for the supporting utilities: GeneformerWrapper + InSilicoKO.
Run with: pytest tests/test_model.py -v

Strategy: everything uses Mocks, no dependency on a real Geneformer model.
Tests logical correctness, not inference quality.
"""

import numpy as np
import pandas as pd
import pytest
import unittest.mock as mock
from pathlib import Path


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

def _make_mock_adata(n_cells=60, n_genes=300, with_cell_type=True):
    import anndata as ad
    import scipy.sparse as sp

    X = sp.random(n_cells, n_genes, density=0.15, format="csr")
    X.data = np.abs(X.data) * 150

    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    gene_names[0] = "PDCD1"
    gene_names[1] = "TOX"
    gene_names[2] = "HAVCR2"

    cell_names = [f"CELL{i:04d}" for i in range(n_cells)]
    adata = ad.AnnData(
        X=X,
        obs={"cell_id": cell_names},
        var={"gene_id": gene_names},
    )
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.raw = adata

    if with_cell_type:
        cell_types = (
            ["CD8_exhausted"] * 25 +
            ["CD8_effector"] * 20 +
            ["Treg"] * 15
        )
        adata.obs["cell_type"] = cell_types

    return adata


def _make_mock_wrapper(gene_names=None):
    """Create a mock GeneformerWrapper in a loaded state."""
    from isp_confound import GeneformerWrapper

    wrapper = GeneformerWrapper.__new__(GeneformerWrapper)
    wrapper.device = "cpu"
    wrapper.max_seq_len = 2048
    wrapper.model_dir = None
    wrapper._model = mock.MagicMock()
    wrapper._tokenizer = mock.MagicMock()

    # build a mock gene vocabulary
    if gene_names is None:
        gene_names = [f"GENE{i:04d}" for i in range(300)]
        gene_names[0] = "PDCD1"
        gene_names[1] = "TOX"
        gene_names[2] = "HAVCR2"

    wrapper._gene2id = {g: i + 1 for i, g in enumerate(gene_names)}
    return wrapper


def _make_mock_ko_result(target_gene="PDCD1", cell_type="CD8_exhausted", n_genes=50):
    """Create a mock KOResult for testing."""
    from isp_confound.insilico_ko import KOResult

    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    mean_shifts = np.random.randn(n_genes).astype(np.float32)
    std_shifts = np.abs(np.random.randn(n_genes).astype(np.float32)) * 0.1
    directions = np.where(mean_shifts > 0, "up", "down")
    eps = 1e-6
    confidence = np.clip(
        1.0 - std_shifts / (np.abs(mean_shifts) + eps), 0.0, 1.0
    )

    de_genes = pd.DataFrame({
        "gene": gene_names,
        "mean_shift": mean_shifts,
        "std_shift": std_shifts,
        "direction": directions,
        "confidence": confidence,
    }).sort_values("mean_shift", key=np.abs, ascending=False).reset_index(drop=True)

    return KOResult(
        target_gene=target_gene,
        cell_type=cell_type,
        n_cells=25,
        de_genes=de_genes,
        mean_shift=mean_shifts,
        std_shift=std_shifts,
        gene_names=gene_names,
    )


# ------------------------------------------------------------------
# GeneformerWrapper tests
# ------------------------------------------------------------------

class TestGeneformerWrapper:
    def test_init_auto_detects_device(self):
        from isp_confound import GeneformerWrapper
        wrapper = GeneformerWrapper()
        assert wrapper.device in ("cuda", "cpu")

    def test_init_respects_device_param(self):
        from isp_confound import GeneformerWrapper
        wrapper = GeneformerWrapper(device="cpu")
        assert wrapper.device == "cpu"

    def test_is_loaded_false_before_load(self):
        from isp_confound import GeneformerWrapper
        wrapper = GeneformerWrapper()
        assert wrapper.is_loaded is False

    def test_is_loaded_true_after_mock_load(self):
        wrapper = _make_mock_wrapper()
        assert wrapper.is_loaded is True

    def test_check_loaded_raises_before_load(self):
        from isp_confound import GeneformerWrapper
        wrapper = GeneformerWrapper()
        with pytest.raises(RuntimeError, match="wrapper.load()"):
            wrapper._check_loaded()

    def test_get_gene2id_returns_dict(self):
        wrapper = _make_mock_wrapper()
        gene2id = wrapper.get_gene2id()
        assert isinstance(gene2id, dict)
        assert len(gene2id) > 0

    def test_get_gene2id_contains_target_genes(self):
        wrapper = _make_mock_wrapper()
        gene2id = wrapper.get_gene2id()
        assert "PDCD1" in gene2id
        assert "TOX" in gene2id

    def test_adata_to_loom_creates_file(self, tmp_path):
        """_adata_to_loom creates a valid loom file (requires loompy)."""
        pytest.importorskip("loompy")
        wrapper = _make_mock_wrapper()
        adata = _make_mock_adata()
        loom_path = wrapper._adata_to_loom(adata, tmp_path, "cell_type")
        assert Path(loom_path).exists()
        assert loom_path.endswith(".loom")

    def test_adata_to_loom_raises_without_loompy(self, tmp_path):
        """Gives a clear error when loompy is not installed."""
        wrapper = _make_mock_wrapper()
        adata = _make_mock_adata()
        with mock.patch.dict("sys.modules", {"loompy": None}):
            with pytest.raises(ImportError, match="loompy"):
                wrapper._adata_to_loom(adata, tmp_path, "cell_type")


# ------------------------------------------------------------------
# KOResult tests
# ------------------------------------------------------------------

class TestKOResult:
    def setup_method(self):
        self.result = _make_mock_ko_result()

    def test_top_upregulated_returns_dataframe(self):
        df = self.result.top_upregulated(n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 5

    def test_top_upregulated_all_up(self):
        df = self.result.top_upregulated(n=10)
        assert (df["direction"] == "up").all()

    def test_top_downregulated_all_down(self):
        df = self.result.top_downregulated(n=10)
        assert (df["direction"] == "down").all()

    def test_high_confidence_threshold(self):
        df = self.result.high_confidence(min_confidence=0.7)
        assert (df["confidence"] >= 0.7).all()

    def test_to_dataframe_returns_copy(self):
        df1 = self.result.to_dataframe()
        df2 = self.result.to_dataframe()
        assert df1 is not df2

    def test_summary_contains_gene_name(self):
        s = self.result.summary()
        assert "PDCD1" in s
        assert "CD8_exhausted" in s

    def test_summary_contains_cell_count(self):
        s = self.result.summary()
        assert "25" in s

    def test_de_genes_has_required_columns(self):
        required = {"gene", "mean_shift", "std_shift", "direction", "confidence"}
        assert required.issubset(set(self.result.de_genes.columns))

    def test_confidence_in_valid_range(self):
        conf = self.result.de_genes["confidence"]
        assert (conf >= 0.0).all() and (conf <= 1.0).all()


# ------------------------------------------------------------------
# InSilicoKO tests
# ------------------------------------------------------------------

class TestInSilicoKO:
    def setup_method(self):
        from isp_confound import InSilicoKO
        self.wrapper = _make_mock_wrapper()
        self.ko = InSilicoKO(self.wrapper, n_mc_samples=3, cell_intoken_size=5)
        self.adata = _make_mock_adata()

    def test_init_stores_params(self):
        assert self.ko.n_mc_samples == 3
        assert self.ko.cell_intoken_size == 5

    def test_predict_raises_for_unknown_gene(self):
        from isp_confound import InSilicoKO
        with pytest.raises(ValueError, match="not in the Geneformer vocabulary"):
            self.ko.predict(self.adata, target_gene="NONEXISTENT_GENE_XYZ")

    def test_predict_raises_for_unknown_cell_type(self):
        with pytest.raises(ValueError, match="does not exist in the data"):
            with mock.patch.object(
                self.ko, "_run_perturber",
                return_value=(np.zeros((2, 10)), [f"G{i}" for i in range(10)])
            ):
                self.ko.predict(
                    self.adata,
                    target_gene="PDCD1",
                    cell_type="NONEXISTENT_TYPE",
                )

    def test_predict_calls_run_perturber(self, tmp_path):
        """predict() correctly calls the underlying _run_perturber."""
        gene_names = [f"GENE{i:04d}" for i in range(20)]
        mock_shifts = np.random.randn(2, 20).astype(np.float32)

        mock_directions = ["up"] * len(gene_names)
        with mock.patch.object(
            self.wrapper, "_adata_to_loom", return_value=str(tmp_path / "fake.loom")
        ):
            with mock.patch.object(self.ko, "_tokenize_loom"):
                with mock.patch.object(
                    self.ko, "_run_perturber",
                    return_value=(mock_shifts, gene_names, mock_directions)
                ) as mock_perturb:
                    result = self.ko.predict(
                        self.adata,
                        target_gene="PDCD1",
                        cell_type="CD8_exhausted",
                        output_dir=tmp_path,
                    )
                    assert mock_perturb.called

        assert result.target_gene == "PDCD1"
        assert result.cell_type == "CD8_exhausted"
        assert isinstance(result, type(result))

    def test_predict_no_cell_type_uses_all(self, tmp_path):
        """cell_type=None uses all cells."""
        gene_names = [f"GENE{i:04d}" for i in range(20)]
        mock_shifts = np.random.randn(2, 20).astype(np.float32)

        mock_directions = ["up"] * len(gene_names)
        with mock.patch.object(
            self.wrapper, "_adata_to_loom", return_value=str(tmp_path / "fake.loom")
        ):
            with mock.patch.object(self.ko, "_tokenize_loom"):
                with mock.patch.object(
                    self.ko, "_run_perturber",
                    return_value=(mock_shifts, gene_names, mock_directions)
                ):
                    result = self.ko.predict(
                        self.adata,
                        target_gene="PDCD1",
                        cell_type=None,
                        output_dir=tmp_path,
                    )

        assert result.cell_type == "all"
        assert result.n_cells == self.adata.n_obs

    def test_predict_batch_skips_unknown_genes(self, tmp_path):
        """Batch prediction skips unknown genes without crashing."""
        gene_names = [f"GENE{i:04d}" for i in range(20)]
        mock_shifts = np.random.randn(2, 20).astype(np.float32)

        mock_directions = ["up"] * len(gene_names)
        with mock.patch.object(
            self.wrapper, "_adata_to_loom", return_value=str(tmp_path / "fake.loom")
        ):
            with mock.patch.object(self.ko, "_tokenize_loom"):
                with mock.patch.object(
                    self.ko, "_run_perturber",
                    return_value=(mock_shifts, gene_names, mock_directions)
                ):
                    results = self.ko.predict_batch(
                        self.adata,
                        target_genes=["PDCD1", "NONEXISTENT_XYZ", "TOX"],
                        cell_type="CD8_exhausted",
                    )

        assert "PDCD1" in results
        assert "TOX" in results
        assert "NONEXISTENT_XYZ" not in results

    def test_compare_cell_types_returns_dict(self, tmp_path):
        """compare_cell_types returns a separate result per cell type."""
        gene_names = [f"GENE{i:04d}" for i in range(20)]

        def mock_perturb(*args, **kwargs):
            shifts = np.random.randn(2, 20).astype(np.float32)
            directions = ["up"] * 20
            return shifts, gene_names, directions

        with mock.patch.object(
            self.wrapper, "_adata_to_loom", return_value=str(tmp_path / "fake.loom")
        ):
            with mock.patch.object(self.ko, "_tokenize_loom"):
                with mock.patch.object(self.ko, "_run_perturber", side_effect=mock_perturb):
                    results = self.ko.compare_cell_types(
                        self.adata,
                        target_gene="PDCD1",
                        cell_types=["CD8_exhausted", "CD8_effector"],
                    )

        assert "CD8_exhausted" in results
        assert "CD8_effector" in results
        assert results["CD8_exhausted"].cell_type == "CD8_exhausted"
        assert results["CD8_effector"].cell_type == "CD8_effector"

    def test_build_result_confidence_range(self):
        """The confidence produced by _build_result is in [0, 1]."""
        gene_names = [f"G{i}" for i in range(30)]
        mean_s = np.random.randn(30).astype(np.float32)
        std_s = np.abs(np.random.randn(30).astype(np.float32)) * 0.1
        shifts = np.stack([mean_s, std_s])

        result = self.ko._build_result(
            target_gene="PDCD1",
            cell_type="CD8_exhausted",
            n_cells=25,
            shifts=shifts,
            gene_names=gene_names,
        )

        conf = result.de_genes["confidence"]
        assert (conf >= 0.0).all() and (conf <= 1.0).all()

    def test_build_result_empty_genes(self):
        """_build_result returns an empty KOResult for an empty gene list without crashing."""
        result = self.ko._build_result(
            target_gene="PDCD1",
            cell_type="CD8_exhausted",
            n_cells=25,
            shifts=np.array([]),
            gene_names=[],
        )
        assert len(result.de_genes) == 0
        assert result.target_gene == "PDCD1"

    def test_parse_perturber_output_csv_mean_shift_col(self):
        pytest.skip("the aggregate_gene_shifts mode no longer uses the mean_shift column")

    def test_parse_perturber_output_csv_shift_col(self):
        pytest.skip("the aggregate_gene_shifts mode no longer uses the shift column")

    def test_parse_perturber_output_pkl_fallback(self, tmp_path):
        """Falls back to a pkl file when there is no CSV."""
        df = pd.DataFrame({
            "Affected_gene_name": ["GENE1", "GENE2", np.nan],
            "Affected": [101, 102, 103],
            "Cosine_sim_mean": [0.9, 0.8, 0.7],
            "Cosine_sim_stdev": [0.01, 0.02, 0.03],
        })
        df.to_pickle(tmp_path / "stats.pkl")

        with mock.patch("isp_confound.insilico_ko._lazy_import") as mock_import:
            mock_stats = mock.MagicMock()
            mock_import.return_value = (mock.MagicMock(InSilicoPerturberStats=mock.MagicMock(return_value=mock_stats)), None)
            shifts, genes, token_ids = self.ko._parse_perturber_output(tmp_path)

        assert genes == ["GENE1", "GENE2"]
        assert token_ids == [101, 102]
        np.testing.assert_allclose(shifts[0], [0.1, 0.2], rtol=1e-6)

    def test_parse_perturber_output_empty_dir(self, tmp_path):
        """An empty directory returns empty arrays without crashing."""
        with mock.patch("isp_confound.insilico_ko._lazy_import") as mock_import:
            mock_import.return_value = (mock.MagicMock(), None)
            shifts, genes, token_ids = self.ko._parse_perturber_output(tmp_path)
        assert len(shifts) == 0
        assert genes == []
        assert token_ids == []
