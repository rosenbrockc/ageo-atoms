"""Tests for ageoa.scipy.sparse_graph atoms."""

import numpy as np
import pytest
import icontract
import scipy.sparse
import scipy.sparse.csgraph

from ageoa.scipy import sparse_graph as ag_gsp


def _make_triangle_graph():
    """Create a simple 3-node triangle graph as a sparse adjacency matrix."""
    W = scipy.sparse.csr_matrix(np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float))
    return W


def _make_path_graph(n=5):
    """Create an n-node path graph as a sparse adjacency matrix."""
    diag = np.ones(n - 1)
    W = scipy.sparse.diags([diag, diag], [-1, 1], shape=(n, n), format="csr").astype(float)
    return W


class TestGraphLaplacian:
    """Tests for the graph_laplacian atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        W = _make_triangle_graph()
        L = ag_gsp.graph_laplacian(W)
        assert L.shape == (3, 3)

    def test_positive_unnormalized(self):
        W = _make_triangle_graph()
        L = ag_gsp.graph_laplacian(W)
        L_dense = L.toarray()
        assert np.allclose(np.diag(L_dense), [2, 2, 2])

    # -- Category 2: Precondition violations --
    def test_require_square(self):
        W = scipy.sparse.csr_matrix(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_gsp.graph_laplacian(W)

    def test_require_symmetric(self):
        W = scipy.sparse.csr_matrix(np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ], dtype=float))
        with pytest.raises(icontract.ViolationError, match="symmetric"):
            ag_gsp.graph_laplacian(W)

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        W = _make_path_graph(5)
        L = ag_gsp.graph_laplacian(W)
        assert L.shape == W.shape

    def test_postcondition_row_sums_zero(self):
        W = _make_triangle_graph()
        L = ag_gsp.graph_laplacian(W)
        row_sums = np.array(L.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 0)

    # -- Category 4: Edge cases --
    def test_single_node(self):
        W = scipy.sparse.csr_matrix(np.array([[0.0]]))
        L = ag_gsp.graph_laplacian(W)
        assert L.shape == (1, 1)
        assert L.toarray()[0, 0] == 0.0

    def test_disconnected_graph(self):
        W = scipy.sparse.csr_matrix(np.zeros((3, 3)))
        L = ag_gsp.graph_laplacian(W)
        assert np.allclose(L.toarray(), 0)

    # -- Category 5: Upstream parity --
    def test_matches_upstream(self):
        W = _make_path_graph(5)
        L_atom = ag_gsp.graph_laplacian(W)
        L_raw = scipy.sparse.csgraph.laplacian(W)
        np.testing.assert_array_almost_equal(L_atom.toarray(), L_raw.toarray())


class TestGraphFourierTransform:
    """Tests for the graph_fourier_transform atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_hat, eigenvalues, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        assert x_hat.shape[0] == 5

    # -- Category 2: Precondition violations --
    def test_require_square(self):
        L = scipy.sparse.csr_matrix(np.ones((3, 4)))
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_gsp.graph_fourier_transform(L, x)

    def test_require_signal_length(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 2.0, 3.0])  # wrong length
        with pytest.raises(icontract.ViolationError, match="Signal length"):
            ag_gsp.graph_fourier_transform(L, x)

    # -- Category 3: Postcondition verification --
    def test_postcondition_returns_tuple(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.ones(5)
        result = ag_gsp.graph_fourier_transform(L, x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_postcondition_coefficient_count(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.ones(5)
        x_hat, eigenvalues, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        assert x_hat.shape[0] == eigenvectors.shape[1]

    # -- Category 4: Edge cases --
    def test_constant_signal(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.ones(5)
        x_hat, eigenvalues, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        # Constant signal should have energy concentrated at eigenvalue 0
        assert eigenvalues[0] < 1e-10

    def test_truncated_k(self):
        W = _make_path_graph(10)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.random.rand(10)
        x_hat, eigenvalues, eigenvectors = ag_gsp.graph_fourier_transform(L, x, k=3)
        assert x_hat.shape[0] == 3

    # -- Category 5: Upstream parity --
    def test_round_trip(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_hat, eigenvalues, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        x_reconstructed = ag_gsp.inverse_graph_fourier_transform(x_hat, eigenvectors)
        assert np.allclose(x_reconstructed, x)


class TestInverseGraphFourierTransform:
    """Tests for the inverse_graph_fourier_transform atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_hat, _, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        result = ag_gsp.inverse_graph_fourier_transform(x_hat, eigenvectors)
        assert np.allclose(result, x)

    # -- Category 2: Precondition violations --
    def test_require_coefficient_count(self):
        x_hat = np.array([1.0, 2.0, 3.0])
        eigenvectors = np.eye(5, 4)  # mismatch: 3 coefficients vs 4 eigenvectors
        with pytest.raises(icontract.ViolationError, match="Coefficient count"):
            ag_gsp.inverse_graph_fourier_transform(x_hat, eigenvectors)

    # -- Category 3: Postcondition verification --
    def test_postcondition_output_length(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.ones(5)
        x_hat, _, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        result = ag_gsp.inverse_graph_fourier_transform(x_hat, eigenvectors)
        assert result.shape[0] == 5

    # -- Category 4: Edge cases --
    def test_zero_coefficients(self):
        eigenvectors = np.eye(3)
        x_hat = np.zeros(3)
        result = ag_gsp.inverse_graph_fourier_transform(x_hat, eigenvectors)
        assert np.allclose(result, 0)

    # -- Category 5: Upstream parity --
    def test_reconstruction_accuracy(self):
        W = _make_path_graph(8)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        rng = np.random.default_rng(42)
        x = rng.standard_normal(8)
        x_hat, _, eigenvectors = ag_gsp.graph_fourier_transform(L, x)
        result = ag_gsp.inverse_graph_fourier_transform(x_hat, eigenvectors)
        np.testing.assert_array_almost_equal(result, x)


class TestHeatKernelDiffusion:
    """Tests for the heat_kernel_diffusion atom."""

    # -- Category 1: Positive path --
    def test_positive_basic(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = ag_gsp.heat_kernel_diffusion(L, x, t=1.0)
        assert result.shape == (5,)

    def test_positive_smoothing(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = ag_gsp.heat_kernel_diffusion(L, x, t=1.0)
        # Diffusion should spread the impulse
        assert result[1] > 0 and result[3] > 0

    # -- Category 2: Precondition violations --
    def test_require_square(self):
        L = scipy.sparse.csr_matrix(np.ones((3, 4)))
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_gsp.heat_kernel_diffusion(L, x, t=1.0)

    def test_require_nonneg_time(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.ones(5)
        with pytest.raises(icontract.ViolationError, match="non-negative"):
            ag_gsp.heat_kernel_diffusion(L, x, t=-1.0)

    def test_require_signal_length(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 2.0])  # wrong length
        with pytest.raises(icontract.ViolationError, match="Signal length"):
            ag_gsp.heat_kernel_diffusion(L, x, t=1.0)

    # -- Category 3: Postcondition verification --
    def test_postcondition_shape(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
        result = ag_gsp.heat_kernel_diffusion(L, x, t=0.5)
        assert result.shape == x.shape

    # -- Category 4: Edge cases --
    def test_zero_time(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ag_gsp.heat_kernel_diffusion(L, x, t=0.0)
        assert np.allclose(result, x)

    def test_large_time_converges(self):
        W = _make_path_graph(5)
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        result = ag_gsp.heat_kernel_diffusion(L, x, t=100.0)
        # With large t, signal should converge toward constant
        assert np.std(result) < np.std(x)

    # -- Category 5: Upstream parity --
    def test_manual_computation(self):
        W = _make_triangle_graph()
        L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(W))
        x = np.array([1.0, 0.0, 0.0])
        t = 0.5
        result = ag_gsp.heat_kernel_diffusion(L, x, t=t)
        # Manually compute: exp(-t*L) @ x via dense eigendecomposition
        L_dense = L.toarray()
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        expected = eigvecs @ (np.exp(-t * eigvals) * (eigvecs.T @ x))
        np.testing.assert_array_almost_equal(result, expected)
