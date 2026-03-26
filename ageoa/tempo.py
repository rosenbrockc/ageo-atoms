"""Tempo.jl wrappers with deterministic contracts and lazy Julia loading."""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Union

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom
from ageoa.tempo_witnesses import witness_offset_tai2tdb, witness_offset_tt2tdb

FloatLike = Union[float, np.float64, int]

def _compute_tt2tdb(seconds: float) -> float:
    """Fairhead & Bretagnon series for TT-TDB offset."""
    T = seconds / (86400.0 * 36525.0)  # Julian centuries from J2000
    M = 6.24006 + 0.017202 * T * 36525.0  # Mean anomaly of Earth
    return 0.001657 * np.sin(M)


def _compute_tai2tt(seconds: float) -> float:
    """TAI to TT offset is a fixed 32.184 seconds."""
    return 32.184


@register_atom(witness_offset_tt2tdb)
@singledispatch
def offset_tt2tdb(seconds: Any) -> Any:
    """Compute the offset from Terrestrial Time (TT) to Barycentric Dynamical Time (TDB) using a Fairhead & Bretagnon series expansion.
    """
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")


@offset_tt2tdb.register(float)
@offset_tt2tdb.register(int)
@offset_tt2tdb.register(np.float64)
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
def _(seconds: FloatLike) -> float:
    return _compute_tt2tdb(float(seconds))


@offset_tt2tdb.register(np.ndarray)
@icontract.ensure(
    lambda result, seconds: isinstance(result, np.ndarray), "result must be numpy array"
)
@icontract.ensure(
    lambda result, seconds: result.shape == seconds.shape, "result shape matches input"
)
def _(seconds: np.ndarray) -> np.ndarray:
    sec_f64 = seconds.astype(np.float64, copy=False)
    T = sec_f64 / (86400.0 * 36525.0)
    M = 6.24006 + 0.017202 * T * 36525.0
    return 0.001657 * np.sin(M)


@register_atom(witness_offset_tai2tdb)
@singledispatch
def offset_tai2tdb(seconds: Any) -> Any:
    """Compute the composite offset from International Atomic Time (TAI) to Barycentric Dynamical Time (TDB) via an intermediate Terrestrial Time (TT) scale.
    """
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")


@offset_tai2tdb.register(float)
@offset_tai2tdb.register(int)
@offset_tai2tdb.register(np.float64)
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
def _(seconds: FloatLike) -> float:
    sec_f64 = float(seconds)
    tai2tt = _compute_tai2tt(sec_f64)
    tt_sec = sec_f64 + tai2tt
    tt2tdb = _compute_tt2tdb(tt_sec)
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
    tai2tt = np.full_like(sec_f64, 32.184)
    tt_sec = sec_f64 + tai2tt
    T = tt_sec / (86400.0 * 36525.0)
    M = 6.24006 + 0.017202 * T * 36525.0
    tt2tdb = 0.001657 * np.sin(M)
    return tai2tt + tt2tdb
