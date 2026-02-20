from typing import Any, Union, List
from functools import singledispatch
import numpy as np
import icontract
from juliacall import Main as jl

# Load the Tempo Julia package
jl.seval("using Tempo")

FloatLike = Union[float, np.float64, int]

@singledispatch
def offset_tt2tdb(seconds: Any) -> Any:
    """Calculate the offset between TT and TDB in seconds.
    
    This function wraps the Tempo.jl `offset_tt2tdb` implementation.
    
    Args:
        seconds: Time since epoch in TT seconds. Can be scalar or numpy array.
        
    Returns:
        The TDB offset in seconds relative to TT.
    """
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")

@offset_tt2tdb.register(float)
@offset_tt2tdb.register(int)
@offset_tt2tdb.register(np.float64)
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
def _(seconds: FloatLike) -> float:
    # Strictly enforce Float64 constraint before passing to Julia backend
    sec_f64 = float(seconds)
    return float(jl.Tempo.offset_tt2tdb(sec_f64))

@offset_tt2tdb.register(np.ndarray)
@icontract.ensure(lambda result, seconds: isinstance(result, np.ndarray), "result must be numpy array")
@icontract.ensure(lambda result, seconds: result.shape == seconds.shape, "result shape matches input")
def _(seconds: np.ndarray) -> np.ndarray:
    # Strictly enforce Float64 constraints
    sec_f64 = seconds.astype(np.float64, copy=False)
    bcast = jl.seval("x -> Tempo.offset_tt2tdb.(x)")
    res = bcast(sec_f64)
    return np.asarray(res, dtype=np.float64)


@singledispatch
def offset_tai2tdb(seconds: Any) -> Any:
    """Calculate the offset from TAI to TDB in seconds.
    
    Args:
        seconds: Time since epoch in TAI seconds. Can be scalar or numpy array.
        
    Returns:
        The TDB offset in seconds relative to TAI.
    """
    raise NotImplementedError(f"Unsupported type: {type(seconds)}")

@offset_tai2tdb.register(float)
@offset_tai2tdb.register(int)
@offset_tai2tdb.register(np.float64)
@icontract.ensure(lambda result: isinstance(result, float), "result must be float")
def _(seconds: FloatLike) -> float:
    # Strictly enforce Float64
    sec_f64 = float(seconds)
    tai2tt = float(jl.Tempo.offset_tai2tt(sec_f64))
    tt_sec = sec_f64 + tai2tt
    tt2tdb = float(jl.Tempo.offset_tt2tdb(tt_sec))
    return tai2tt + tt2tdb

@offset_tai2tdb.register(np.ndarray)
@icontract.ensure(lambda result, seconds: isinstance(result, np.ndarray), "result must be numpy array")
@icontract.ensure(lambda result, seconds: result.shape == seconds.shape, "result shape matches input")
def _(seconds: np.ndarray) -> np.ndarray:
    sec_f64 = seconds.astype(np.float64, copy=False)
    bcast_tai2tt = jl.seval("x -> Tempo.offset_tai2tt.(x)")
    bcast_tt2tdb = jl.seval("x -> Tempo.offset_tt2tdb.(x)")
    
    tai2tt = np.asarray(bcast_tai2tt(sec_f64), dtype=np.float64)
    tt_sec = sec_f64 + tai2tt
    tt2tdb = np.asarray(bcast_tt2tdb(tt_sec), dtype=np.float64)
    return tai2tt + tt2tdb
