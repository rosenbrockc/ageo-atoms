"""Tempo.jl wrappers with deterministic contracts and lazy Julia loading."""

from __future__ import annotations

from typing import Union

import icontract
import numpy as np

from ageoa.ghost.registry import register_atom
from ageoa.tempo_witnesses import witness_offset_tai2tdb, witness_offset_tt2tdb

FloatLike = Union[float, np.float64, int]
SecondsLike = Union[FloatLike, np.ndarray]


def _is_numeric_array(seconds: object) -> bool:
    return isinstance(seconds, np.ndarray) and np.issubdtype(seconds.dtype, np.number)

def _compute_tt2tdb(seconds: float) -> float:
    """Fairhead & Bretagnon series for TT-TDB offset."""
    T = seconds / (86400.0 * 36525.0)  # Julian centuries from J2000
    M = 6.24006 + 0.017202 * T * 36525.0  # Mean anomaly of Earth
    return 0.001657 * np.sin(M)


def _compute_tai2tt(seconds: float) -> float:
    """TAI to TT offset is a fixed 32.184 seconds."""
    return 32.184


@register_atom(witness_offset_tt2tdb)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def offset_tt2tdb(seconds: SecondsLike) -> float | np.ndarray:
    """Compute the offset from Terrestrial Time (TT) to Barycentric Dynamical Time (TDB) using a Fairhead & Bretagnon series expansion.
    """
    if isinstance(seconds, np.ndarray):
        if not _is_numeric_array(seconds):
            raise NotImplementedError(f"Unsupported dtype: {seconds.dtype}")
        sec_f64 = seconds.astype(np.float64, copy=False)
        T = sec_f64 / (86400.0 * 36525.0)
        M = 6.24006 + 0.017202 * T * 36525.0
        return 0.001657 * np.sin(M)
    if isinstance(seconds, (float, int, np.float64)):
        return _compute_tt2tdb(float(seconds))
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")


@register_atom(witness_offset_tai2tdb)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def offset_tai2tdb(seconds: SecondsLike) -> float | np.ndarray:
    """Compute the composite offset from International Atomic Time (TAI) to Barycentric Dynamical Time (TDB) via an intermediate Terrestrial Time (TT) scale.
    """
    if isinstance(seconds, np.ndarray):
        if not _is_numeric_array(seconds):
            raise NotImplementedError(f"Unsupported dtype: {seconds.dtype}")
        sec_f64 = seconds.astype(np.float64, copy=False)
        tai2tt = np.full_like(sec_f64, 32.184)
        tt_sec = sec_f64 + tai2tt
        T = tt_sec / (86400.0 * 36525.0)
        M = 6.24006 + 0.017202 * T * 36525.0
        tt2tdb = 0.001657 * np.sin(M)
        return tai2tt + tt2tdb
    if isinstance(seconds, (float, int, np.float64)):
        sec_f64 = float(seconds)
        tai2tt = _compute_tai2tt(sec_f64)
        tt_sec = sec_f64 + tai2tt
        tt2tdb = _compute_tt2tdb(tt_sec)
        return tai2tt + tt2tdb
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")
