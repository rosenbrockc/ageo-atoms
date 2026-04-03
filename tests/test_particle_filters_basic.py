from __future__ import annotations

import numpy as np
import pytest

from auditlib.runtime_probes import safe_import_module

_abstract = safe_import_module("ageoa.ghost.abstract")
_atoms = safe_import_module("ageoa.particle_filters.basic.atoms")
_witnesses = safe_import_module("ageoa.particle_filters.basic.witnesses")

AbstractArray = _abstract.AbstractArray
filter_step_preparation_and_dispatch = _atoms.filter_step_preparation_and_dispatch
witness_filter_step_preparation_and_dispatch = _witnesses.witness_filter_step_preparation_and_dispatch


def test_filter_step_preparation_and_dispatch_returns_canonical_bundle() -> None:
    prior_state = {
        "particles": np.array([1.0, 2.0], dtype=float),
        "weights": np.array([0.4, 0.6], dtype=float),
        "rng_seed": 7,
    }
    model_spec = {"transition": "unit"}
    control_t = np.array([0.5], dtype=float)
    observation_t = np.array([1.5], dtype=float)

    result = filter_step_preparation_and_dispatch(prior_state, model_spec, control_t, observation_t)

    assert isinstance(result, tuple)
    assert len(result) == 5
    returned_prior_state, returned_model_spec, returned_control_t, returned_observation_t, rng_key = result
    assert returned_prior_state is prior_state
    assert returned_model_spec is model_spec
    np.testing.assert_array_equal(returned_control_t, control_t)
    np.testing.assert_array_equal(returned_observation_t, observation_t)
    np.testing.assert_array_equal(rng_key, np.array([7], dtype=np.int64))


def test_filter_step_preparation_and_dispatch_requires_explicit_rng_seed() -> None:
    with pytest.raises(KeyError, match="rng_seed"):
        filter_step_preparation_and_dispatch(
            {"particles": np.array([1.0], dtype=float), "weights": np.array([1.0], dtype=float)},
            {"transition": "unit"},
            np.array([0.5], dtype=float),
            np.array([1.5], dtype=float),
        )


def test_witness_filter_step_preparation_and_dispatch_returns_bundle_metadata() -> None:
    prior_state = AbstractArray(shape=(2,), dtype="float64")
    model_spec = AbstractArray(shape=(), dtype="object")
    control_t = AbstractArray(shape=(1,), dtype="float64")
    observation_t = AbstractArray(shape=(1,), dtype="float64")

    result = witness_filter_step_preparation_and_dispatch(prior_state, model_spec, control_t, observation_t)

    assert isinstance(result, tuple)
    assert len(result) == 5
    returned_prior_state, returned_model_spec, returned_control_t, returned_observation_t, rng_key = result
    assert returned_prior_state.shape == prior_state.shape
    assert returned_model_spec.shape == model_spec.shape
    assert returned_control_t.shape == control_t.shape
    assert returned_observation_t.shape == observation_t.shape
    assert rng_key.shape == (1,)
    assert rng_key.dtype == "int64"
