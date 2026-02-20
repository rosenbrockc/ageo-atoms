"""Tempo.jl wrappers with deterministic contracts and lazy Julia loading."""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Union

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom
from ageoa.tempo_witnesses import witness_offset_tai2tdb, witness_offset_tt2tdb

FloatLike = Union[float, np.float64, int]

_jl: Any | None = None
_tempo_loaded = False
_offset_tt2tdb_bcast: Any | None = None
_offset_tai2tt_bcast: Any | None = None


def _get_jl() -> Any:
    """Import juliacall lazily and load Tempo once."""
    global _jl, _tempo_loaded
    if _jl is None:
        from juliacall import Main as jl_main

        _jl = jl_main
    if not _tempo_loaded:
        _jl.seval("using Tempo")
        _tempo_loaded = True
    return _jl


def _get_tt2tdb_bcast() -> Any:
    global _offset_tt2tdb_bcast
    if _offset_tt2tdb_bcast is None:
        _offset_tt2tdb_bcast = _get_jl().seval("x -> Tempo.offset_tt2tdb.(x)")
    return _offset_tt2tdb_bcast


def _get_tai2tt_bcast() -> Any:
    global _offset_tai2tt_bcast
    if _offset_tai2tt_bcast is None:
        _offset_tai2tt_bcast = _get_jl().seval("x -> Tempo.offset_tai2tt.(x)")
    return _offset_tai2tt_bcast


@register_atom(witness_offset_tt2tdb)
@singledispatch
def offset_tt2tdb(seconds: Any) -> Any:
    """Compute the offset between two related time coordinate systems using a Fairhead & Bretagnon series expansion.

    <!-- conceptual_profile
    {
        "abstract_name": "Relational Temporal Coordinate Transformer",
        "conceptual_transform": "Computes the relative offset between two related time coordinate scales at specified points in time. It resolves relativistic and gravitational effects to align local measurement frames with a global reference frame.",
        "abstract_inputs": [
            {
                "name": "seconds",
                "description": "A scalar or tensor representing coordinates in the source temporal frame."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The computed offset values required to align with the target temporal frame."
            }
        ],
        "algorithmic_properties": [
            "coordinate-transform",
            "relativistic-alignment",
            "deterministic",
            "vectorized-capable"
        ],
        "cross_disciplinary_applications": [
            "Aligning measurements taken in different reference frames in distributed sensor networks.",
            "Correcting for propagation-delay differences between synchronized clock domains.",
            "Converting between coordinate time representations in relativistic simulation frameworks."
        ]
    }
    /conceptual_profile -->
    """
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")


@offset_tt2tdb.register(float)
@offset_tt2tdb.register(int)
@offset_tt2tdb.register(np.float64)
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
def _(seconds: FloatLike) -> float:
    sec_f64 = float(seconds)
    jl = _get_jl()
    return float(jl.Tempo.offset_tt2tdb(sec_f64))


@offset_tt2tdb.register(np.ndarray)
@icontract.ensure(
    lambda result, seconds: isinstance(result, np.ndarray), "result must be numpy array"
)
@icontract.ensure(
    lambda result, seconds: result.shape == seconds.shape, "result shape matches input"
)
def _(seconds: np.ndarray) -> np.ndarray:
    sec_f64 = seconds.astype(np.float64, copy=False)
    bcast = _get_tt2tdb_bcast()
    res = bcast(sec_f64)
    return np.asarray(res, dtype=np.float64)


@register_atom(witness_offset_tai2tdb)
@singledispatch
def offset_tai2tdb(seconds: Any) -> Any:
    """Compute the composite offset between two time coordinate systems via an intermediate scale.

    <!-- conceptual_profile
    {
        "abstract_name": "Composite Multi-Scale Temporal Resolver",
        "conceptual_transform": "Calculates the cumulative offset between two time coordinate systems via a fixed intermediate offset and a series-expansion correction. It performs a series of nested coordinate transformations to resolve the total temporal displacement between a local reference and a global dynamical reference.",
        "abstract_inputs": [
            {
                "name": "seconds",
                "description": "A scalar or tensor of coordinates in the initial temporal frame."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The resolved cumulative offset values."
            }
        ],
        "algorithmic_properties": [
            "nested-transform",
            "cumulative-offset-resolution",
            "deterministic",
            "vectorized-capable"
        ],
        "cross_disciplinary_applications": [
            "Synchronizing distributed clock references across heterogeneous timing domains.",
            "Resolving micro-second timing differences in distributed particle physics experiments.",
            "Standardizing high-precision timestamps for international metrology."
        ]
    }
    /conceptual_profile -->
    """
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")


@offset_tai2tdb.register(float)
@offset_tai2tdb.register(int)
@offset_tai2tdb.register(np.float64)
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
def _(seconds: FloatLike) -> float:
    sec_f64 = float(seconds)
    jl = _get_jl()
    tai2tt = float(jl.Tempo.offset_tai2tt(sec_f64))
    tt_sec = sec_f64 + tai2tt
    tt2tdb = float(jl.Tempo.offset_tt2tdb(tt_sec))
    return tai2tt + tt2tdb


@offset_tai2tdb.register(np.ndarray)
@icontract.ensure(
    lambda result, seconds: isinstance(result, np.ndarray), "result must be numpy array"
)
@icontract.ensure(
    lambda result, seconds: result.shape == seconds.shape, "result shape matches input"
)
def _(seconds: np.ndarray) -> np.ndarray:
    sec_f64 = seconds.astype(np.float64, copy=False)
    bcast_tai2tt = _get_tai2tt_bcast()
    bcast_tt2tdb = _get_tt2tdb_bcast()

    tai2tt = np.asarray(bcast_tai2tt(sec_f64), dtype=np.float64)
    tt_sec = sec_f64 + tai2tt
    tt2tdb = np.asarray(bcast_tt2tdb(tt_sec), dtype=np.float64)
    return tai2tt + tt2tdb
