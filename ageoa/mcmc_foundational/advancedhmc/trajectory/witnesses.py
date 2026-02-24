"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations

import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore

try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_show(io: AbstractArray, s: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_slicets(s: AbstractArray, H0: AbstractArray, zcand: AbstractArray) -> AbstractArray:
    """Ghost witness for Slicets."""
    result = AbstractArray(
        shape=s.shape,
        dtype="float64",
    )
    return result

def witness_multinomialts(s: AbstractArray, H0: AbstractArray, zcand: AbstractArray) -> AbstractArray:
    """Ghost witness for Multinomialts."""
    result = AbstractArray(
        shape=s.shape,
        dtype="float64",
    )
    return result

def witness_combine(rng: AbstractArray, s1: AbstractArray, s2: AbstractArray) -> AbstractArray:
    """Ghost witness for Combine."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_combine(zcand: AbstractArray, s1: AbstractArray, s2: AbstractArray) -> AbstractArray:
    """Ghost witness for Combine."""
    result = AbstractArray(
        shape=zcand.shape,
        dtype="float64",
    )
    return result

def witness_combine(rng: AbstractArray, s1: AbstractArray, s2: AbstractArray) -> AbstractArray:
    """Ghost witness for Combine."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_combine(zcand: AbstractArray, s1: AbstractArray, s2: AbstractArray) -> AbstractArray:
    """Ghost witness for Combine."""
    result = AbstractArray(
        shape=zcand.shape,
        dtype="float64",
    )
    return result

def witness_mh_accept(rng: AbstractArray, s: AbstractArray, s′: AbstractArray) -> AbstractArray:
    """Ghost witness for Mh Accept."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, τ: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_transition(h: AbstractArray, τ: AbstractArray, z: AbstractArray) -> AbstractArray:
    """Ghost witness for Transition."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_transition(rng: AbstractArray, h: AbstractArray, τ: AbstractArray, z: AbstractArray) -> AbstractArray:
    """Ghost witness for Transition."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_accept_phasepoint!(z: AbstractArray, z′: AbstractArray, is_accept: AbstractArray) -> AbstractArray:
    """Ghost witness for Accept Phasepoint!."""
    result = AbstractArray(
        shape=z.shape,
        dtype="float64",
    )
    return result

def witness_accept_phasepoint!(z: AbstractArray, z′: AbstractArray, is_accept: AbstractArray) -> AbstractArray:
    """Ghost witness for Accept Phasepoint!."""
    result = AbstractArray(
        shape=z.shape,
        dtype="float64",
    )
    return result

def witness_sample_phasepoint(rng: AbstractArray, τ: AbstractArray, h: AbstractArray, z: AbstractArray) -> AbstractArray:
    """Ghost witness for Sample Phasepoint."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_randcat(rng: AbstractArray, zs: AbstractArray, unnorm_ℓp: AbstractArray) -> AbstractArray:
    """Ghost witness for Randcat."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_randcat(rng: AbstractArray, zs: AbstractArray, unnorm_ℓP: AbstractArray) -> AbstractArray:
    """Ghost witness for Randcat."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_sample_phasepoint(rng: AbstractArray, τ: AbstractArray, h: AbstractArray, z: AbstractArray) -> AbstractArray:
    """Ghost witness for Sample Phasepoint."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_generalisednouturn(tc: AbstractArray) -> AbstractArray:
    """Ghost witness for Generalisednouturn."""
    result = AbstractArray(
        shape=tc.shape,
        dtype="float64",
    )
    return result

def witness_turnstatistic(z: AbstractArray) -> AbstractArray:
    """Ghost witness for Turnstatistic."""
    result = AbstractArray(
        shape=z.shape,
        dtype="float64",
    )
    return result

def witness_show(io: AbstractArray, d: AbstractArray) -> AbstractArray:
    """Ghost witness for Show."""
    result = AbstractArray(
        shape=io.shape,
        dtype="float64",
    )
    return result

def witness_termination(s: AbstractArray, nt: AbstractArray, H0: AbstractArray, H′: AbstractArray) -> AbstractArray:
    """Ghost witness for Termination."""
    result = AbstractArray(
        shape=s.shape,
        dtype="float64",
    )
    return result

def witness_termination(s: AbstractArray, nt: AbstractArray, H0: AbstractArray, H′: AbstractArray) -> AbstractArray:
    """Ghost witness for Termination."""
    result = AbstractArray(
        shape=s.shape,
        dtype="float64",
    )
    return result

def witness_isterminated(h: AbstractArray, t: AbstractArray) -> AbstractArray:
    """Ghost witness for Isterminated."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_isterminated(h: AbstractArray, t: AbstractArray) -> AbstractArray:
    """Ghost witness for Isterminated."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_isterminated(tc: AbstractArray, h: AbstractArray, t: AbstractArray, tleft: AbstractArray, tright: AbstractArray) -> AbstractArray:
    """Ghost witness for Isterminated."""
    result = AbstractArray(
        shape=tc.shape,
        dtype="float64",
    )
    return result

def witness_check_left_subtree(h: AbstractArray, t: AbstractArray, tleft: AbstractArray, tright: AbstractArray) -> AbstractArray:
    """Ghost witness for Check Left Subtree."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_check_right_subtree(h: AbstractArray, t: AbstractArray, tleft: AbstractArray, tright: AbstractArray) -> AbstractArray:
    """Ghost witness for Check Right Subtree."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_generalised_uturn_criterion(rho: AbstractArray, p_sharp_minus: AbstractArray, p_sharp_plus: AbstractArray) -> AbstractArray:
    """Ghost witness for Generalised Uturn Criterion."""
    result = AbstractArray(
        shape=rho.shape,
        dtype="float64",
    )
    return result

def witness_isterminated(tc: AbstractArray, h: AbstractArray, t: AbstractArray, _tleft: AbstractArray, _tright: AbstractArray) -> AbstractArray:
    """Ghost witness for Isterminated."""
    result = AbstractArray(
        shape=tc.shape,
        dtype="float64",
    )
    return result

def witness_build_tree(rng: AbstractArray, nt: AbstractArray, h: AbstractArray, z: AbstractArray, sampler: AbstractArray, v: AbstractArray, j: AbstractArray, H0: AbstractArray) -> AbstractArray:
    """Ghost witness for Build Tree."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_transition(rng: AbstractArray, h: AbstractArray, τ: AbstractArray, z0: AbstractArray) -> AbstractArray:
    """Ghost witness for Transition."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_a(h: AbstractArray, z: AbstractArray, ϵ: AbstractArray) -> AbstractArray:
    """Ghost witness for A."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_find_good_stepsize(rng: AbstractArray, h: AbstractArray, θ: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Good Stepsize."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_find_good_stepsize(h: AbstractArray, θ: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Good Stepsize."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_mh_accept_ratio(rng: AbstractArray, Horiginal: AbstractArray, Hproposal: AbstractArray) -> AbstractArray:
    """Ghost witness for Mh Accept Ratio."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result

def witness_mh_accept_ratio(rng: AbstractArray, Horiginal: AbstractArray, Hproposal: AbstractArray) -> AbstractArray:
    """Ghost witness for Mh Accept Ratio."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",
    )
    return result
