"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, s: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_slicets)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda H0: H0 is not None, "H0 cannot be None")
@icontract.require(lambda zcand: zcand is not None, "zcand cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Slicets output must not be None")
def slicets(s: Any, H0: Any, zcand: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_multinomialts)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda H0: H0 is not None, "H0 cannot be None")
@icontract.require(lambda zcand: zcand is not None, "zcand cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Multinomialts output must not be None")
def multinomialts(s: Any, H0: Any, zcand: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_combine)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda s1: s1 is not None, "s1 cannot be None")
@icontract.require(lambda s2: s2 is not None, "s2 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Combine output must not be None")
def combine(rng: Any, s1: Any, s2: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_combine)
@icontract.require(lambda zcand: zcand is not None, "zcand cannot be None")
@icontract.require(lambda s1: s1 is not None, "s1 cannot be None")
@icontract.require(lambda s2: s2 is not None, "s2 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Combine output must not be None")
def combine(zcand: Any, s1: Any, s2: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_combine)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda s1: s1 is not None, "s1 cannot be None")
@icontract.require(lambda s2: s2 is not None, "s2 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Combine output must not be None")
def combine(rng: Any, s1: Any, s2: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_combine)
@icontract.require(lambda zcand: zcand is not None, "zcand cannot be None")
@icontract.require(lambda s1: s1 is not None, "s1 cannot be None")
@icontract.require(lambda s2: s2 is not None, "s2 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Combine output must not be None")
def combine(zcand: Any, s1: Any, s2: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_mh_accept)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda s′: s′ is not None, "s′ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Mh Accept output must not be None")
def mh_accept(rng: Any, s: Any, s′: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda τ: τ is not None, "τ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, τ: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_transition)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda τ: τ is not None, "τ cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Transition output must not be None")
def transition(h: Any, τ: Any, z: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_transition)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda τ: τ is not None, "τ cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Transition output must not be None")
def transition(rng: Any, h: Any, τ: Any, z: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_accept_phasepoint!)
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda z′: z′ is not None, "z′ cannot be None")
@icontract.require(lambda is_accept: is_accept is not None, "is_accept cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Accept Phasepoint! output must not be None")
def accept_phasepoint!(z: Any, z′: Any, is_accept: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_accept_phasepoint!)
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda z′: z′ is not None, "z′ cannot be None")
@icontract.require(lambda is_accept: is_accept is not None, "is_accept cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Accept Phasepoint! output must not be None")
def accept_phasepoint!(z: Any, z′: Any, is_accept: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_sample_phasepoint)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda τ: τ is not None, "τ cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Sample Phasepoint output must not be None")
def sample_phasepoint(rng: Any, τ: Any, h: Any, z: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_randcat)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda zs: zs is not None, "zs cannot be None")
@icontract.require(lambda unnorm_ℓp: unnorm_ℓp is not None, "unnorm_ℓp cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Randcat output must not be None")
def randcat(rng: Any, zs: Any, unnorm_ℓp: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_randcat)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda zs: zs is not None, "zs cannot be None")
@icontract.require(lambda unnorm_ℓP: unnorm_ℓP is not None, "unnorm_ℓP cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Randcat output must not be None")
def randcat(rng: Any, zs: Any, unnorm_ℓP: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_sample_phasepoint)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda τ: τ is not None, "τ cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Sample Phasepoint output must not be None")
def sample_phasepoint(rng: Any, τ: Any, h: Any, z: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_generalisednouturn)
@icontract.require(lambda tc: tc is not None, "tc cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Generalisednouturn output must not be None")
def generalisednouturn(tc: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_turnstatistic)
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Turnstatistic output must not be None")
def turnstatistic(z: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Show output must not be None")
def show(io: Any, d: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_termination)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda nt: nt is not None, "nt cannot be None")
@icontract.require(lambda H0: H0 is not None, "H0 cannot be None")
@icontract.require(lambda H′: H′ is not None, "H′ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Termination output must not be None")
def termination(s: Any, nt: Any, H0: Any, H′: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_termination)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda nt: nt is not None, "nt cannot be None")
@icontract.require(lambda H0: H0 is not None, "H0 cannot be None")
@icontract.require(lambda H′: H′ is not None, "H′ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Termination output must not be None")
def termination(s: Any, nt: Any, H0: Any, H′: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_isterminated)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Isterminated output must not be None")
def isterminated(h: Any, t: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_isterminated)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Isterminated output must not be None")
def isterminated(h: Any, t: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_isterminated)
@icontract.ensure(lambda result, **kwargs: result is not None, "Isterminated output must not be None")
def isterminated(tc: Any, h: Any, t: Any, tleft: Any, tright: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_check_left_subtree)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.require(lambda tleft: tleft is not None, "tleft cannot be None")
@icontract.require(lambda tright: tright is not None, "tright cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Check Left Subtree output must not be None")
def check_left_subtree(h: Any, t: Any, tleft: Any, tright: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_check_right_subtree)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.require(lambda tleft: tleft is not None, "tleft cannot be None")
@icontract.require(lambda tright: tright is not None, "tright cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Check Right Subtree output must not be None")
def check_right_subtree(h: Any, t: Any, tleft: Any, tright: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_generalised_uturn_criterion)
@icontract.require(lambda rho: rho is not None, "rho cannot be None")
@icontract.require(lambda p_sharp_minus: p_sharp_minus is not None, "p_sharp_minus cannot be None")
@icontract.require(lambda p_sharp_plus: p_sharp_plus is not None, "p_sharp_plus cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Generalised Uturn Criterion output must not be None")
def generalised_uturn_criterion(rho: Any, p_sharp_minus: Any, p_sharp_plus: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_isterminated)
@icontract.ensure(lambda result, **kwargs: result is not None, "Isterminated output must not be None")
def isterminated(tc: Any, h: Any, t: Any, _tleft: Any, _tright: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_build_tree)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda nt: nt is not None, "nt cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda sampler: sampler is not None, "sampler cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.require(lambda j: j is not None, "j cannot be None")
@icontract.require(lambda H0: H0 is not None, "H0 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Build Tree output must not be None")
def build_tree(rng: Any, nt: Any, h: Any, z: Any, sampler: Any, v: Any, j: Any, H0: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_transition)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda τ: τ is not None, "τ cannot be None")
@icontract.require(lambda z0: z0 is not None, "z0 cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Transition output must not be None")
def transition(rng: Any, h: Any, τ: Any, z0: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_a)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda ϵ: ϵ is not None, "ϵ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "A output must not be None")
def a(h: Any, z: Any, ϵ: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_find_good_stepsize)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda θ: θ is not None, "θ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Find Good Stepsize output must not be None")
def find_good_stepsize(rng: Any, h: Any, θ: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_find_good_stepsize)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda θ: θ is not None, "θ cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Find Good Stepsize output must not be None")
def find_good_stepsize(h: Any, θ: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_mh_accept_ratio)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda Horiginal: Horiginal is not None, "Horiginal cannot be None")
@icontract.require(lambda Hproposal: Hproposal is not None, "Hproposal cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Mh Accept Ratio output must not be None")
def mh_accept_ratio(rng: Any, Horiginal: Any, Hproposal: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_mh_accept_ratio)
@icontract.require(lambda rng: rng is not None, "rng cannot be None")
@icontract.require(lambda Horiginal: Horiginal is not None, "Horiginal cannot be None")
@icontract.require(lambda Hproposal: Hproposal is not None, "Hproposal cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Mh Accept Ratio output must not be None")
def mh_accept_ratio(rng: Any, Horiginal: Any, Hproposal: Any) -> Any:
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def show_ffi(io, s):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, s)")

def slicets_ffi(s, H0, zcand):
    """FFI bridge to Julia implementation of Slicets."""
    return jl.eval("slicets(s, H0, zcand)")

def multinomialts_ffi(s, H0, zcand):
    """FFI bridge to Julia implementation of Multinomialts."""
    return jl.eval("multinomialts(s, H0, zcand)")

def combine_ffi(rng, s1, s2):
    """FFI bridge to Julia implementation of Combine."""
    return jl.eval("combine(rng, s1, s2)")

def combine_ffi(zcand, s1, s2):
    """FFI bridge to Julia implementation of Combine."""
    return jl.eval("combine(zcand, s1, s2)")

def combine_ffi(rng, s1, s2):
    """FFI bridge to Julia implementation of Combine."""
    return jl.eval("combine(rng, s1, s2)")

def combine_ffi(zcand, s1, s2):
    """FFI bridge to Julia implementation of Combine."""
    return jl.eval("combine(zcand, s1, s2)")

def mh_accept_ffi(rng, s, s′):
    """FFI bridge to Julia implementation of Mh Accept."""
    return jl.eval("mh_accept(rng, s, s′)")

def show_ffi(io, τ):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, τ)")

def transition_ffi(h, τ, z):
    """FFI bridge to Julia implementation of Transition."""
    return jl.eval("transition(h, τ, z)")

def transition_ffi(rng, h, τ, z):
    """FFI bridge to Julia implementation of Transition."""
    return jl.eval("transition(rng, h, τ, z)")

def accept_phasepoint!_ffi(z, z′, is_accept):
    """FFI bridge to Julia implementation of Accept Phasepoint!."""
    return jl.eval("accept_phasepoint!(z, z′, is_accept)")

def accept_phasepoint!_ffi(z, z′, is_accept):
    """FFI bridge to Julia implementation of Accept Phasepoint!."""
    return jl.eval("accept_phasepoint!(z, z′, is_accept)")

def sample_phasepoint_ffi(rng, τ, h, z):
    """FFI bridge to Julia implementation of Sample Phasepoint."""
    return jl.eval("sample_phasepoint(rng, τ, h, z)")

def randcat_ffi(rng, zs, unnorm_ℓp):
    """FFI bridge to Julia implementation of Randcat."""
    return jl.eval("randcat(rng, zs, unnorm_ℓp)")

def randcat_ffi(rng, zs, unnorm_ℓP):
    """FFI bridge to Julia implementation of Randcat."""
    return jl.eval("randcat(rng, zs, unnorm_ℓP)")

def sample_phasepoint_ffi(rng, τ, h, z):
    """FFI bridge to Julia implementation of Sample Phasepoint."""
    return jl.eval("sample_phasepoint(rng, τ, h, z)")

def generalisednouturn_ffi(tc):
    """FFI bridge to Julia implementation of Generalisednouturn."""
    return jl.eval("generalisednouturn(tc)")

def turnstatistic_ffi(z):
    """FFI bridge to Julia implementation of Turnstatistic."""
    return jl.eval("turnstatistic(z)")

def show_ffi(io, d):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, d)")

def termination_ffi(s, nt, H0, H′):
    """FFI bridge to Julia implementation of Termination."""
    return jl.eval("termination(s, nt, H0, H′)")

def termination_ffi(s, nt, H0, H′):
    """FFI bridge to Julia implementation of Termination."""
    return jl.eval("termination(s, nt, H0, H′)")

def isterminated_ffi(h, t):
    """FFI bridge to Julia implementation of Isterminated."""
    return jl.eval("isterminated(h, t)")

def isterminated_ffi(h, t):
    """FFI bridge to Julia implementation of Isterminated."""
    return jl.eval("isterminated(h, t)")

def isterminated_ffi(tc, h, t, tleft, tright):
    """FFI bridge to Julia implementation of Isterminated."""
    return jl.eval("isterminated(tc, h, t, tleft, tright)")

def check_left_subtree_ffi(h, t, tleft, tright):
    """FFI bridge to Julia implementation of Check Left Subtree."""
    return jl.eval("check_left_subtree(h, t, tleft, tright)")

def check_right_subtree_ffi(h, t, tleft, tright):
    """FFI bridge to Julia implementation of Check Right Subtree."""
    return jl.eval("check_right_subtree(h, t, tleft, tright)")

def generalised_uturn_criterion_ffi(rho, p_sharp_minus, p_sharp_plus):
    """FFI bridge to Julia implementation of Generalised Uturn Criterion."""
    return jl.eval("generalised_uturn_criterion(rho, p_sharp_minus, p_sharp_plus)")

def isterminated_ffi(tc, h, t, _tleft, _tright):
    """FFI bridge to Julia implementation of Isterminated."""
    return jl.eval("isterminated(tc, h, t, _tleft, _tright)")

def build_tree_ffi(rng, nt, h, z, sampler, v, j, H0):
    """FFI bridge to Julia implementation of Build Tree."""
    return jl.eval("build_tree(rng, nt, h, z, sampler, v, j, H0)")

def transition_ffi(rng, h, τ, z0):
    """FFI bridge to Julia implementation of Transition."""
    return jl.eval("transition(rng, h, τ, z0)")

def a_ffi(h, z, ϵ):
    """FFI bridge to Julia implementation of A."""
    return jl.eval("a(h, z, ϵ)")

def find_good_stepsize_ffi(rng, h, θ):
    """FFI bridge to Julia implementation of Find Good Stepsize."""
    return jl.eval("find_good_stepsize(rng, h, θ)")

def find_good_stepsize_ffi(h, θ):
    """FFI bridge to Julia implementation of Find Good Stepsize."""
    return jl.eval("find_good_stepsize(h, θ)")

def mh_accept_ratio_ffi(rng, Horiginal, Hproposal):
    """FFI bridge to Julia implementation of Mh Accept Ratio."""
    return jl.eval("mh_accept_ratio(rng, Horiginal, Hproposal)")

def mh_accept_ratio_ffi(rng, Horiginal, Hproposal):
    """FFI bridge to Julia implementation of Mh Accept Ratio."""
    return jl.eval("mh_accept_ratio(rng, Horiginal, Hproposal)")
