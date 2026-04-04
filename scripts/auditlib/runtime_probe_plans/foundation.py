"""Foundational family probe plans split from the monolithic runtime probe registry."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def _assert_array(expected: np.ndarray, *, atol: float = 1e-8):
    def _validator(result: Any) -> None:
        np.testing.assert_allclose(np.asarray(result), expected, atol=atol)

    return _validator


def _assert_sparse_shape(expected_shape: tuple[int, int]):
    def _validator(result: Any) -> None:
        assert sp.issparse(result)
        assert tuple(result.shape) == expected_shape

    return _validator


def _assert_shape(expected_shape: tuple[int, ...]):
    def _validator(result: Any) -> None:
        assert tuple(np.asarray(result).shape) == expected_shape

    return _validator


def _skyfield_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    return {
        "ageoa.skyfield.calculate_vector_angle": ProbePlan(
            positive=ProbeCase(
                "Computes the angle between orthogonal basis vectors",
                lambda func: func(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
                lambda result: np.testing.assert_allclose(result, np.pi / 2.0, atol=1e-12),
            ),
            negative=ProbeCase(
                "reject missing second vector",
                lambda func: func(np.array([1.0, 0.0, 0.0]), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.skyfield.compute_spherical_coordinate_rates": ProbePlan(
            positive=ProbeCase(
                "Converts a Cartesian state into spherical coordinates and rates",
                lambda func: func(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])),
                lambda result: (
                    isinstance(result, tuple)
                    and len(result) == 6
                    and np.testing.assert_allclose(
                        np.asarray(result, dtype=float),
                        np.array(
                            [
                                3.7416573867739413,
                                0.9302740141154721,
                                1.1071487177940904,
                                0.3741657386773941,
                                0.0,
                                0.0,
                            ]
                        ),
                        atol=1e-12,
                    )
                ),
            ),
            negative=ProbeCase(
                "reject missing velocity vector",
                lambda func: func(np.array([1.0, 2.0, 3.0]), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _mint_attention_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    def _assert_numpy_attention_result(result: Any) -> None:
        arr = np.asarray(result, dtype=float)
        assert arr.shape == (2, 2)
        assert np.all(np.isfinite(arr))

    def _assert_torch_attention_result(result: Any) -> None:
        import torch

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, attn = result
        assert isinstance(output, torch.Tensor)
        assert isinstance(attn, torch.Tensor)
        assert tuple(output.shape) == (2, 2)
        assert tuple(attn.shape) == (2, 2)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    return {
        "ageoa.mint.axial_attention.rowselfattention": ProbePlan(
            positive=ProbeCase(
                "NumPy row self-attention returns a finite same-shape output",
                lambda func: func(
                    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
                    np.ones((2, 2), dtype=int),
                    np.zeros(2, dtype=int),
                ),
                _assert_numpy_attention_result,
            ),
            negative=ProbeCase(
                "reject a missing attention tensor",
                lambda func: func(None, np.ones((2, 2), dtype=int), np.zeros(2, dtype=int)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.axial_attention.row_self_attention": ProbePlan(
            positive=ProbeCase(
                "Torch row self-attention returns output and attention matrices with normalized rows",
                lambda func: func(
                    __import__("torch").tensor([[1.0, 0.0], [0.0, 1.0]], dtype=__import__("torch").float32),
                    __import__("torch").ones((2, 2), dtype=__import__("torch").bool),
                    __import__("torch").zeros(2, dtype=__import__("torch").bool),
                ),
                _assert_torch_attention_result,
            ),
            negative=ProbeCase(
                "reject a missing attention tensor",
                lambda func: func(
                    None,
                    __import__("torch").ones((2, 2), dtype=__import__("torch").bool),
                    __import__("torch").zeros(2, dtype=__import__("torch").bool),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _mint_top_level_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    def _assert_axial_attention_result(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        output, attn_probs = result
        assert tuple(np.asarray(output).shape) == (2, 2, 1, 4)
        assert tuple(np.asarray(attn_probs).shape) == (1, 2, 2)
        assert np.all(np.isfinite(np.asarray(output, dtype=float)))
        assert np.all(np.isfinite(np.asarray(attn_probs, dtype=float)))

    def _assert_rotary_result(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        q_out, k_out = result
        assert tuple(np.asarray(q_out).shape) == (1, 2, 4)
        assert tuple(np.asarray(k_out).shape) == (1, 2, 4)
        assert np.all(np.isfinite(np.asarray(q_out, dtype=float)))
        assert np.all(np.isfinite(np.asarray(k_out, dtype=float)))

    return {
        "ageoa.mint.axial_attention": ProbePlan(
            positive=ProbeCase(
                "mint axial attention returns contextualized embeddings and attention weights for a tiny 4D tensor",
                lambda func: func(np.arange(16, dtype=np.float64).reshape(2, 2, 1, 4)),
                _assert_axial_attention_result,
            ),
            negative=ProbeCase(
                "reject an attention tensor with the wrong rank",
                lambda func: func(np.arange(8, dtype=np.float64).reshape(2, 4)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.rotary_positional_embeddings": ProbePlan(
            positive=ProbeCase(
                "mint rotary positional embeddings rotate a small query/key pair without changing shape",
                lambda func: func(
                    np.arange(8, dtype=np.float64).reshape(1, 2, 4),
                    np.arange(8, dtype=np.float64).reshape(1, 2, 4) + 1.0,
                ),
                _assert_rotary_result,
            ),
            negative=ProbeCase(
                "reject a query tensor with an odd embedding dimension",
                lambda func: func(
                    np.arange(6, dtype=np.float64).reshape(1, 2, 3),
                    np.arange(6, dtype=np.float64).reshape(1, 2, 3),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _e2e_ppg_reconstruction_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    def _assert_windowed_reconstruction(result: Any) -> None:
        arr = np.asarray(result, dtype=float)
        assert arr.shape == (400,)
        assert np.all(np.isfinite(arr))

    return {
        "ageoa.e2e_ppg.reconstruction.windowed_signal_reconstruction": ProbePlan(
            positive=ProbeCase(
                "reconstruction returns a same-length signal for an all-clean windowed input",
                lambda func: func(
                    np.sin(np.linspace(0.0, 8.0 * np.pi, 400, dtype=float)),
                    [[i for i in range(400)]],
                    [],
                    20,
                    False,
                ),
                _assert_windowed_reconstruction,
            ),
            negative=ProbeCase(
                "reject a missing signal array",
                lambda func: func(None, [[0, 1, 2]], [], 20, False),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _e2e_ppg_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    signal = np.sin(np.linspace(0.0, 12.0 * np.pi, 1000, dtype=float))
    heart_cycles = [
        np.sin(np.linspace(0.0, np.pi, 40, dtype=float)),
        np.sin(np.linspace(0.1, np.pi + 0.1, 40, dtype=float)),
        np.sin(np.linspace(-0.1, np.pi - 0.1, 40, dtype=float)),
    ]

    def _assert_peak_indices(result: Any) -> None:
        arr = np.asarray(result, dtype=int)
        assert arr.ndim == 1
        assert np.all(arr >= 0)
        assert np.all(np.diff(arr) >= 0)

    def _assert_ppg_reconstruction(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 3
        reconstructed, clean_indices, noisy_indices = result
        assert np.asarray(reconstructed).shape == (signal.shape[0],)
        assert isinstance(clean_indices, list)
        assert isinstance(noisy_indices, list)

    def _assert_ppg_sqa(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_indices, noisy_indices = result
        assert isinstance(clean_indices, list)
        assert isinstance(noisy_indices, list)

    def _assert_template_features(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        euclidean, corr = result
        assert np.isfinite(float(euclidean))
        assert np.isfinite(float(corr))

    return {
        "ageoa.e2e_ppg.kazemi_peak_detection": ProbePlan(
            positive=ProbeCase(
                "kazemi peak detection returns monotonic peak indices on a synthetic pulse trace",
                lambda func: func(signal, 20, 4, 1, 1),
                _assert_peak_indices,
            ),
            negative=ProbeCase(
                "reject a missing signal array",
                lambda func: func(None, 20, 4, 1, 1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.e2e_ppg.ppg_reconstruction": ProbePlan(
            positive=ProbeCase(
                "ppg reconstruction returns a reconstructed trace and grouped indices",
                lambda func: func(signal, [[i for i in range(700)]], [[i for i in range(700, 1000)]], 20),
                _assert_ppg_reconstruction,
            ),
            negative=ProbeCase(
                "reject a missing signal array",
                lambda func: func(None, [], [], 20),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.e2e_ppg.ppg_sqa": ProbePlan(
            positive=ProbeCase(
                "ppg sqa returns clean/noisy grouped indices for a synthetic signal",
                lambda func: func(signal, 20),
                _assert_ppg_sqa,
            ),
            negative=ProbeCase(
                "reject a missing signal array",
                lambda func: func(None, 20),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.e2e_ppg.kazemi_wrapper_d12.normalizesignal": ProbePlan(
            positive=ProbeCase(
                "kazemi wrapper normalization rescales a signal to the unit interval",
                lambda func: func(np.array([1.0, 3.0, 2.0], dtype=float)),
                _assert_shape((3,)),
            ),
            negative=ProbeCase(
                "reject a missing normalization array",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.e2e_ppg.kazemi_wrapper_d12.wrapperevaluate": ProbePlan(
            positive=ProbeCase(
                "kazemi wrapper evaluation returns monotonic peak indices",
                lambda func: func(
                    np.array([0.1] * 10 + [1.0] + [0.1] * 19, dtype=float),
                    np.array([1.0] * 30, dtype=float),
                ),
                _assert_peak_indices,
            ),
            negative=ProbeCase(
                "reject a missing prediction array",
                lambda func: func(None, np.array([1.0] * 30, dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.numpy.random_v2.continuousmultivariatesampler": ProbePlan(
            positive=ProbeCase(
                "numpy random_v2 continuous multivariate sampler returns paired sample arrays",
                lambda func: func(
                    np.array([0.0, 1.0], dtype=float),
                    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
                    np.array([1.0, 2.0], dtype=float),
                    size=2,
                ),
                lambda result: (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and tuple(np.asarray(result[0]).shape) == (2, 2)
                    and tuple(np.asarray(result[1]).shape) == (2, 2)
                ),
            ),
            negative=ProbeCase(
                "reject a missing mean vector",
                lambda func: func(None, np.eye(2), np.array([1.0, 2.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.numpy.random_v2.discreteeventsampler": ProbePlan(
            positive=ProbeCase(
                "numpy random_v2 discrete event sampler returns multinomial counts",
                lambda func: func(5, np.array([0.2, 0.8], dtype=float)),
                lambda result: (
                    tuple(np.asarray(result).shape) == (2,)
                    and int(np.asarray(result).sum()) == 5
                ),
            ),
            negative=ProbeCase(
                "reject a missing probability vector",
                lambda func: func(5, None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.numpy.random_v2.combinatoricssampler": ProbePlan(
            positive=ProbeCase(
                "numpy random_v2 combinatorics sampler returns a permutation and choice sample",
                lambda func: func(np.array([1, 2, 3], dtype=int), np.array([10, 20, 30], dtype=int), size=2, replace=False),
                lambda result: (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and sorted(np.asarray(result[0]).tolist()) == [1, 2, 3]
                    and tuple(np.asarray(result[1]).shape) == (2,)
                ),
            ),
            negative=ProbeCase(
                "reject a missing permutation input",
                lambda func: func(None, np.array([10, 20, 30], dtype=int)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.jFOF.find_fof_clusters": ProbePlan(
            positive=ProbeCase(
                "jFOF wrapper returns one integer cluster label per point on a simple periodic point cloud",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 0.0],
                            [0.1, 0.0],
                            [1.0, 1.0],
                        ],
                        dtype=float,
                    ),
                    0.2,
                    2.0,
                ),
                lambda result: (
                    tuple(np.asarray(result).shape) == (3,)
                    and np.asarray(result).dtype.kind in {"i", "u"}
                    and np.asarray(result)[0] == np.asarray(result)[1]
                ),
            ),
            negative=ProbeCase(
                "reject a missing point cloud",
                lambda func: func(None, 0.2, 2.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.e2e_ppg.template_matching.templatefeaturecomputation": ProbePlan(
            positive=ProbeCase(
                "template matching computes deterministic euclidean/correlation features",
                lambda func: func(heart_cycles),
                _assert_template_features,
            ),
            negative=ProbeCase(
                "reject a missing heart-cycle list",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _datadriven_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    def _assert_equation_result(result: Any) -> None:
        assert type(result).__name__ == "EquationResult"
        assert isinstance(result.equations, list)
        assert len(result.equations) >= 1
        assert all(isinstance(eq, str) and eq.strip() for eq in result.equations)
        assert isinstance(result.parameter_map, dict)

    return {
        "ageoa.datadriven.discover_equations": ProbePlan(
            positive=ProbeCase(
                "Discovers at least one sparse equation from a one-feature linear system",
                lambda func: func(
                    np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float),
                    np.array([[0.0, 2.0, 4.0, 6.0]], dtype=float),
                    ["x"],
                    max_degree=1,
                    lambda_val=0.1,
                ),
                _assert_equation_result,
            ),
            negative=ProbeCase(
                "reject invalid Julia identifiers in variable_names",
                lambda func: func(
                    np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float),
                    np.array([[0.0, 2.0, 4.0, 6.0]], dtype=float),
                    ["1bad"],
                    max_degree=1,
                    lambda_val=0.1,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _alphafold_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    def _assert_alpha_nodes(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        nodes, state = result
        assert tuple(np.asarray(nodes).shape) == (4, 8)
        assert hasattr(state, "nodes")
        assert tuple(np.asarray(state.nodes).shape) == (4, 8)

    def _assert_alpha_frames(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        frames, state = result
        assert tuple(np.asarray(frames).shape) == (4, 7)
        assert hasattr(state, "frames")
        assert tuple(np.asarray(state.frames).shape) == (4, 7)

    def _assert_alpha_coords(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        coords, state = result
        assert tuple(np.asarray(coords).shape) == (4, 37, 3)
        assert hasattr(state, "frames")

    def _state():
        from ageoa.alphafold.state_models import AlphaFoldStructuralState

        return AlphaFoldStructuralState()

    return {
        "ageoa.alphafold.invariant_point_attention": ProbePlan(
            positive=ProbeCase(
                "alphafold invariant point attention updates node embeddings on a tiny synthetic residue set",
                lambda func: func(
                    np.zeros((4, 8), dtype=float),
                    np.zeros((4, 4, 3), dtype=float),
                    np.zeros((4, 7), dtype=float),
                    _state(),
                ),
                _assert_alpha_nodes,
            ),
            negative=ProbeCase(
                "reject pair features with mismatched sequence dimensions",
                lambda func: func(
                    np.zeros((4, 8), dtype=float),
                    np.zeros((5, 4, 3), dtype=float),
                    np.zeros((4, 7), dtype=float),
                    _state(),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.alphafold.equivariant_frame_update": ProbePlan(
            positive=ProbeCase(
                "alphafold frame update returns the same frame shape on a tiny synthetic residue set",
                lambda func: func(
                    np.zeros((4, 7), dtype=float),
                    np.zeros((4, 8), dtype=float),
                    _state(),
                ),
                _assert_alpha_frames,
            ),
            negative=ProbeCase(
                "reject a scalar node tensor",
                lambda func: func(
                    np.zeros((4, 7), dtype=float),
                    np.array(1.0),
                    _state(),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.alphafold.coordinate_reconstruction": ProbePlan(
            positive=ProbeCase(
                "alphafold coordinate reconstruction returns residue coordinates from tiny torsion inputs",
                lambda func: func(
                    np.zeros((4, 7), dtype=float),
                    np.zeros((4, 7, 2), dtype=float),
                    _state(),
                ),
                _assert_alpha_coords,
            ),
            negative=ProbeCase(
                "reject torsions without sin/cos pairs",
                lambda func: func(
                    np.zeros((4, 7), dtype=float),
                    np.zeros((4, 7, 3), dtype=float),
                    _state(),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _pronto_backlash_filter_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    state = np.array([0.5, 1.0, 0.0, 0.0], dtype=np.float64)
    return {
        "ageoa.pronto.backlash_filter.initializebacklashfilterstate": ProbePlan(
            positive=ProbeCase(
                "initialize a local backlash filter state snapshot",
                lambda fn: fn(),
                _assert_array(state),
            ),
            negative=ProbeCase(
                "reject unexpected arguments for the zero-parameter initializer",
                lambda fn: fn(unexpected=True),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.backlash_filter.updatealphaparameter": ProbePlan(
            positive=ProbeCase(
                "update only the alpha slot of the local backlash filter state",
                lambda fn: fn(state.copy(), 0.25),
                _assert_array(np.array([0.25, 1.0, 0.0, 0.0], dtype=np.float64)),
            ),
            negative=ProbeCase(
                "reject non-array state input for alpha updates",
                lambda fn: fn({"state": "bad"}, 0.25),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.pronto.backlash_filter.updatecrossingtimemaximum": ProbePlan(
            positive=ProbeCase(
                "update only the crossing-time slot of the local backlash filter state",
                lambda fn: fn(state.copy(), 2.5),
                _assert_array(np.array([0.5, 2.5, 0.0, 0.0], dtype=np.float64)),
            ),
            negative=ProbeCase(
                "reject non-finite crossing-time updates",
                lambda fn: fn(state.copy(), np.nan),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _sklearn_image_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    image = np.arange(16, dtype=float).reshape(4, 4)
    volume = np.arange(8, dtype=float).reshape(2, 2, 2)
    return {
        "ageoa.sklearn.images.extract_patches_2d": ProbePlan(
            positive=ProbeCase(
                "extract 2x2 patches from a 4x4 image",
                lambda func: func(image, (2, 2)),
                _assert_shape((9, 2, 2)),
            ),
            negative=ProbeCase(
                "reject patch sizes larger than the image",
                lambda func: func(image, (5, 5)),
                expect_exception=True,
            ),
        ),
        "ageoa.sklearn.images.reconstruct_from_patches_2d": ProbePlan(
            positive=ProbeCase(
                "reconstruct a 4x4 image from extracted 2x2 patches",
                lambda func: func(np.arange(36, dtype=float).reshape(9, 2, 2), (4, 4)),
                _assert_shape((4, 4)),
            ),
            negative=ProbeCase(
                "reject incompatible patch layouts",
                lambda func: func(np.arange(16, dtype=float).reshape(4, 2, 2), (4, 4)),
                expect_exception=True,
            ),
        ),
        "ageoa.sklearn.images.img_to_graph": ProbePlan(
            positive=ProbeCase(
                "build an image graph for a 2x2x2 volume",
                lambda func: func(volume),
                _assert_sparse_shape((8, 8)),
            ),
            negative=ProbeCase(
                "reject a scalar input instead of an image volume",
                lambda func: func(np.array(1.0)),
                expect_exception=True,
            ),
        ),
        "ageoa.sklearn.images.grid_to_graph": ProbePlan(
            positive=ProbeCase(
                "build a voxel grid graph for a 2x2x2 lattice",
                lambda func: func(2, 2, 2),
                _assert_sparse_shape((8, 8)),
            ),
            negative=ProbeCase(
                "reject a zero-sized grid dimension",
                lambda func: func(0, 2, 2),
                expect_exception=True,
            ),
        ),
    }


def _scipy_interpolate_v2_plans(ProbeCase: type, ProbePlan: type) -> dict[str, Any]:
    def _assert_cubic_spline_callable(result: Any) -> None:
        values = np.asarray(result(np.array([0.5, 1.5], dtype=float)))
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))

    def _assert_rbf_callable(result: Any) -> None:
        values = np.asarray(result(np.array([[0.5], [1.5]], dtype=float)))
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))

    return {
        "ageoa.scipy.interpolate_v2.cubicsplinefit": ProbePlan(
            positive=ProbeCase(
                "build a callable cubic spline interpolator for three 1D samples",
                lambda func: func(
                    np.array([0.0, 1.0, 2.0], dtype=float),
                    np.array([0.0, 1.0, 0.0], dtype=float),
                ),
                _assert_cubic_spline_callable,
            ),
            negative=ProbeCase(
                "reject non-monotonic x coordinates",
                lambda func: func(
                    np.array([0.0, 2.0, 1.0], dtype=float),
                    np.array([0.0, 1.0, 0.0], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.interpolate_v2.rbfinterpolatorfit": ProbePlan(
            positive=ProbeCase(
                "build a callable radial basis interpolator for simple 1D scattered data",
                lambda func: func(
                    np.array([[0.0], [1.0], [2.0]], dtype=float),
                    np.array([0.0, 1.0, 0.0], dtype=float),
                    smoothing=0.0,
                    kernel="linear",
                ),
                _assert_rbf_callable,
            ),
            negative=ProbeCase(
                "reject a missing data-value array",
                lambda func: func(np.array([[0.0], [1.0]], dtype=float), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def get_probe_plans() -> dict[str, Any]:
    """Return the foundation-family probe registry."""
    from ..runtime_probes import ProbeCase, ProbePlan

    plans: dict[str, Any] = {}
    plans.update(_skyfield_plans(ProbeCase, ProbePlan))
    plans.update(_mint_attention_plans(ProbeCase, ProbePlan))
    plans.update(_mint_top_level_plans(ProbeCase, ProbePlan))
    plans.update(_e2e_ppg_plans(ProbeCase, ProbePlan))
    plans.update(_e2e_ppg_reconstruction_plans(ProbeCase, ProbePlan))
    plans.update(_datadriven_plans(ProbeCase, ProbePlan))
    plans.update(_alphafold_plans(ProbeCase, ProbePlan))
    plans.update(_pronto_backlash_filter_plans(ProbeCase, ProbePlan))
    plans.update(_scipy_interpolate_v2_plans(ProbeCase, ProbePlan))
    plans.update(_sklearn_image_plans(ProbeCase, ProbePlan))
    return plans
