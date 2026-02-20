import os

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import icontract

from ageoa.ghost.registry import register_atom
from ageoa.ghost.witnesses import (
    witness_graph_laplacian, witness_graph_fourier_transform,
    witness_heat_kernel_diffusion,
)

_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"


def _is_symmetric(m: scipy.sparse.spmatrix, atol: float = 1e-10) -> bool:
    """Check that a sparse matrix is symmetric within tolerance."""
    if m.shape[0] != m.shape[1]:
        return False
    diff = m - m.T
    if diff.nnz == 0:
        return True
    return bool(np.all(np.abs(diff.data) < atol))


def _is_square_sparse(m: scipy.sparse.spmatrix) -> bool:
    """Check that a sparse matrix is square."""
    return m.shape[0] == m.shape[1]


def _eigenvalues_nonneg(L: scipy.sparse.spmatrix, k: int = 1) -> bool:
    """Check that the smallest eigenvalue of L is >= -epsilon (PSD check)."""
    n = L.shape[0]
    if n < 3:
        L_dense = L.toarray()
        eigvals = np.linalg.eigvalsh(L_dense)
        return bool(np.all(eigvals >= -1e-8))
    k_check = min(k, n - 2)
    eigvals = scipy.sparse.linalg.eigsh(L, k=k_check, which="SM", return_eigenvectors=False)
    return bool(np.all(eigvals >= -1e-8))


def _total_variation(L: scipy.sparse.spmatrix, x: np.ndarray) -> float:
    """Compute the total variation x^T L x of signal x on graph with Laplacian L."""
    Lx = L.dot(x)
    return float(x.dot(Lx))


@register_atom(witness_graph_laplacian)
@icontract.require(lambda W: _is_symmetric(W), "Weight matrix W must be symmetric")
@icontract.require(lambda W: _is_square_sparse(W), "Weight matrix W must be square")
@icontract.ensure(lambda result, W: result.shape == W.shape, "Laplacian shape must match input shape")
@icontract.ensure(
    lambda result: _eigenvalues_nonneg(result, k=1),
    "Graph Laplacian must be positive semi-definite",
    enabled=_SLOW_CHECKS,
)
def graph_laplacian(
    W: scipy.sparse.spmatrix,
    normed: bool = False,
    return_diag: bool = False,
) -> scipy.sparse.spmatrix:
    """Compute the graph Laplacian of a weighted adjacency matrix.

    Computes L = D - W (unnormalized) or the normalized Laplacian
    from the symmetric weight matrix W.

    Args:
        W: Symmetric sparse weight/adjacency matrix of shape (n, n).
        normed: If True, compute the normalized Laplacian.
        return_diag: If True, also return the diagonal. The atom
            returns only the Laplacian; set to False.

    Returns:
        The graph Laplacian as a sparse matrix of shape (n, n).
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Relational Divergence Operator Generator",
        "conceptual_transform": "Computes the discrete Laplacian operator for a weighted relational system. It encodes the local connectivity and weight distribution into a second-order differential matrix, capturing the divergence-like properties of the graph topology.",
        "abstract_inputs": [
            {
                "name": "W",
                "description": "A symmetric sparse tensor representing the weighted connectivity (adjacency) of the system."
            },
            {
                "name": "normed",
                "description": "A boolean indicating whether to normalize the operator by the local connectivity density."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A sparse square tensor representing the graph Laplacian operator."
            }
        ],
        "algorithmic_properties": [
            "differential-operator",
            "topological-encoding",
            "positive-semi-definite"
        ],
        "cross_disciplinary_applications": [
            "Analyzing the vibrational modes of a molecular structure.",
            "Quantifying the robustness of a communications network topology.",
            "Computing the structural equilibrium of a multi-component truss system."
        ]
    }
    <!-- /conceptual_profile -->
    """
    result = scipy.sparse.csgraph.laplacian(W, normed=normed, return_diag=return_diag)
    if return_diag:
        return result[0]
    return result


@register_atom(witness_graph_fourier_transform)
@icontract.require(lambda L: _is_square_sparse(L), "Laplacian L must be square")
@icontract.require(lambda L, x: x.shape[0] == L.shape[0], "Signal length must equal graph size")
@icontract.ensure(
    lambda result: isinstance(result, tuple) and len(result) == 3,
    "Must return (x_hat, eigenvalues, eigenvectors)",
)
@icontract.ensure(
    lambda result, L: result[0].shape[0] == min(result[0].shape[0], L.shape[0]),
    "Coefficient count must be consistent with graph size",
)
def graph_fourier_transform(
    L: scipy.sparse.spmatrix,
    x: np.ndarray,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Graph Fourier Transform of a signal on a graph.

    Projects the signal x onto the eigenvectors of the graph Laplacian L.
    The GFT generalizes the classical DFT to irregular graph domains.

    Args:
        L: Graph Laplacian, sparse matrix of shape (n, n).
        x: Graph signal of length n.
        k: Number of eigenvectors to use. If None, uses all n.

    Returns:
        Tuple of (x_hat, eigenvalues, eigenvectors) where x_hat are the
        GFT coefficients, eigenvalues are the graph frequencies, and
        eigenvectors are the GFT basis vectors.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Irregular Domain Spectral Projector",
        "conceptual_transform": "Generalizes spectral analysis to irregular relational domains by projecting signals onto the intrinsic basis functions (eigenvectors) of the system's topological operator. It reveals the 'frequency' components of a signal as they relate to the underlying connectivity.",
        "abstract_inputs": [
            {
                "name": "L",
                "description": "A square tensor representing the topological divergence operator (Laplacian)."
            },
            {
                "name": "x",
                "description": "A 1D tensor representing a signal defined on the nodes of the graph."
            },
            {
                "name": "k",
                "description": "An optional integer specifying the number of lower-frequency components to compute."
            }
        ],
        "abstract_outputs": [
            {
                "name": "x_hat",
                "description": "A 1D tensor of spectral amplitudes corresponding to the graph frequencies."
            },
            {
                "name": "eigenvalues",
                "description": "A 1D tensor representing the graph frequencies (eigenvalues)."
            },
            {
                "name": "eigenvectors",
                "description": "A 2D tensor representing the spectral basis functions."
            }
        ],
        "algorithmic_properties": [
            "spectral-projection",
            "domain-generalized",
            "basis-transformation"
        ],
        "cross_disciplinary_applications": [
            "Analyzing the spread of influence in a social network spectral domain.",
            "Detecting localized structural anomalies in a complex mesh geometry.",
            "Compressing signal data defined on irregular sensor network topologies."
        ]
    }
    <!-- /conceptual_profile -->
    """
    n = L.shape[0]
    if k is None or k >= n - 1:
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    else:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k, which="SM")
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    x_hat = eigenvectors.T @ x
    return x_hat, eigenvalues, eigenvectors


@icontract.require(lambda x_hat, eigenvectors: x_hat.shape[0] == eigenvectors.shape[1], "Coefficient count must match number of eigenvectors")
@icontract.ensure(lambda result, eigenvectors: result.shape[0] == eigenvectors.shape[0], "Output length must equal graph size")
def inverse_graph_fourier_transform(
    x_hat: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Compute the Inverse Graph Fourier Transform.

    Reconstructs the graph signal from GFT coefficients and eigenvectors.

    Args:
        x_hat: GFT coefficients of length k.
        eigenvectors: GFT basis vectors of shape (n, k).

    Returns:
        Reconstructed graph signal of length n.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Irregular Domain Spectral Synthesizer",
        "conceptual_transform": "Reconstructs a signal on an irregular relational domain from its spectral representation. It performs a linear combination of topological basis functions weighted by their spectral amplitudes.",
        "abstract_inputs": [
            {
                "name": "x_hat",
                "description": "A 1D tensor of spectral amplitudes."
            },
            {
                "name": "eigenvectors",
                "description": "A 2D tensor representing the topological basis functions."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The reconstructed signal in the original relational domain."
            }
        ],
        "algorithmic_properties": [
            "spectral-synthesis",
            "linear-reconstruction",
            "basis-combination"
        ],
        "cross_disciplinary_applications": [
            "Synthesizing smooth fields over complex 3D mesh surfaces.",
            "Reconstructing missing sensor values in an irregular network from spectral priors.",
            "Visualizing low-frequency trends in a high-dimensional relational dataset."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return eigenvectors @ x_hat


@register_atom(witness_heat_kernel_diffusion)
@icontract.require(lambda L: _is_square_sparse(L), "Laplacian L must be square")
@icontract.require(lambda t: t >= 0, "Diffusion time t must be non-negative")
@icontract.require(lambda L, x: x.shape[0] == L.shape[0], "Signal length must equal graph size")
@icontract.ensure(lambda result, x: result.shape == x.shape, "Output shape must be preserved")
@icontract.ensure(
    lambda result, L, x: _total_variation(L, result) <= _total_variation(L, x) + 1e-8,
    "Heat diffusion must reduce total variation (smoothing)",
    enabled=_SLOW_CHECKS,
)
def heat_kernel_diffusion(
    L: scipy.sparse.spmatrix,
    x: np.ndarray,
    t: float,
    k: int | None = None,
) -> np.ndarray:
    """Apply heat kernel diffusion to a graph signal.

    Computes exp(-t*L) @ x, which smooths the signal x over the graph
    topology. The diffusion reduces the total variation of the signal.

    Args:
        L: Graph Laplacian, sparse matrix of shape (n, n).
        x: Graph signal of length n.
        t: Diffusion time parameter. Must be >= 0. Larger values
            produce smoother outputs.
        k: Number of eigenvectors to use for the approximation.
            If None, uses all n eigenvectors.

    Returns:
        The diffused graph signal of length n.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Relational Low-Pass Smoothing Transformer",
        "conceptual_transform": "Simulates a diffusion process (heat flow) over a relational topology to smooth local signal fluctuations while preserving the underlying structure. It acts as a low-pass filter in the graph spectral domain, exponentially suppressing high-frequency components over time.",
        "abstract_inputs": [
            {
                "name": "L",
                "description": "A square tensor representing the topological divergence operator."
            },
            {
                "name": "x",
                "description": "A 1D tensor representing the initial signal distribution."
            },
            {
                "name": "t",
                "description": "A non-negative scalar representing the diffusion time (smoothing magnitude)."
            },
            {
                "name": "k",
                "description": "An optional integer for low-rank approximation of the diffusion process."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The diffused (smoothed) signal distribution."
            }
        ],
        "algorithmic_properties": [
            "diffusion-based",
            "low-pass-filtering",
            "variation-reducing",
            "topological-smoothing"
        ],
        "cross_disciplinary_applications": [
            "Smoothing noisy measurements over a geographic sensor network.",
            "Modeling the propagation of influence or perturbation through a networked population of agents.",
            "Extracting stable features from a noisy 3D mesh for shape recognition."
        ]
    }
    <!-- /conceptual_profile -->
    """
    n = L.shape[0]
    if k is None or k >= n - 1:
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    else:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k, which="SM")
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    x_hat = eigenvectors.T @ x
    heat_filter = np.exp(-t * eigenvalues)
    x_hat_filtered = heat_filter * x_hat
    return eigenvectors @ x_hat_filtered
