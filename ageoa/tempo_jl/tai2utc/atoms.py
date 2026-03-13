from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_cal2jd, witness_calhms2jd, witness_fd2hms, witness_fd2hmsf, witness_find_day, witness_find_dayinyear, witness_find_month, witness_find_year, witness_hms2fd, witness_isleapyear, witness_jd2cal, witness_jd2calhms, witness_lastj2000dayofyear, witness_tai2utc, witness_utc2tai

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_isleapyear)
@icontract.require(lambda year: year is not None, "year cannot be None")
@icontract.ensure(lambda result: result is not None, "Isleapyear output must not be None")
def isleapyear(year: int) -> bool:
    """Isleapyear.

    Args:
        year (int): Description.

    Returns:
        bool: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_find_dayinyear)
@icontract.require(lambda month: month is not None, "month cannot be None")
@icontract.require(lambda day: day is not None, "day cannot be None")
@icontract.require(lambda isleap: isleap is not None, "isleap cannot be None")
@icontract.ensure(lambda result: result is not None, "Find Dayinyear output must not be None")
def find_dayinyear(month: int, day: int, isleap: bool) -> int:
    """Find dayinyear.

    Args:
        month (int): Description.
        day (int): Description.
        isleap (bool): Description.

    Returns:
        int: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_find_year)
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.ensure(lambda result: result is not None, "Find Year output must not be None")
def find_year(d: float) -> int:
    """Find year.

    Args:
        d (float): Description.

    Returns:
        int: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_find_month)
@icontract.require(lambda dayinyear: dayinyear is not None, "dayinyear cannot be None")
@icontract.require(lambda isleap: isleap is not None, "isleap cannot be None")
@icontract.ensure(lambda result: result is not None, "Find Month output must not be None")
def find_month(dayinyear: int, isleap: bool) -> int:
    """Find month.

    Args:
        dayinyear (int): Description.
        isleap (bool): Description.

    Returns:
        int: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_find_day)
@icontract.require(lambda dayinyear: dayinyear is not None, "dayinyear cannot be None")
@icontract.require(lambda month: month is not None, "month cannot be None")
@icontract.require(lambda isleap: isleap is not None, "isleap cannot be None")
@icontract.ensure(lambda result: result is not None, "Find Day output must not be None")
def find_day(dayinyear: int, month: int, isleap: bool) -> int:
    """Find day.

    Args:
        dayinyear (int): Description.
        month (int): Description.
        isleap (bool): Description.

    Returns:
        int: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_lastj2000dayofyear)
@icontract.require(lambda year: year is not None, "year cannot be None")
@icontract.ensure(lambda result: result is not None, "Lastj2000Dayofyear output must not be None")
def lastj2000dayofyear(year: int) -> int:
    """Lastj2000dayofyear.

    Args:
        year (int): Description.

    Returns:
        int: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_hms2fd)
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda m: m is not None, "m cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.ensure(lambda result: result is not None, "Hms2Fd output must not be None")
def hms2fd(h: int, m: int, s: float) -> float:
    """Hms2fd.

    Args:
        h (int): Description.
        m (int): Description.
        s (float): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_fd2hms)
@icontract.require(lambda fd: fd is not None, "fd cannot be None")
@icontract.ensure(lambda result: result is not None, "Fd2Hms output must not be None")
def fd2hms(fd: float) -> tuple[int, int, float]:
    """Fd2hms.

    Args:
        fd (float): Description.

    Returns:
        tuple[int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_fd2hmsf)
@icontract.require(lambda fd: fd is not None, "fd cannot be None")
@icontract.ensure(lambda result: result is not None, "Fd2Hmsf output must not be None")
def fd2hmsf(fd: float) -> tuple[int, int, int, float]:
    """Fd2hmsf.

    Args:
        fd (float): Description.

    Returns:
        tuple[int, int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_cal2jd)
@icontract.require(lambda Y: Y is not None, "Y cannot be None")
@icontract.require(lambda M: M is not None, "M cannot be None")
@icontract.require(lambda D: D is not None, "D cannot be None")
@icontract.ensure(lambda result: result is not None, "Cal2Jd output must not be None")
def cal2jd(Y: int, M: int, D: int) -> float:
    """Cal2jd.

    Args:
        Y (int): Description.
        M (int): Description.
        D (int): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_calhms2jd)
@icontract.require(lambda Y, M, D: 1 <= M <= 12 and 1 <= D <= 31, "M must be 1-12, D must be 1-31")
@icontract.ensure(lambda result: result is not None, "Calhms2Jd output must not be None")
def calhms2jd(Y: int, M: int, D: int, h: int, m: int, sec: float) -> float:
    """Calhms2jd.

    Args:
        Y (int): Description.
        M (int): Description.
        D (int): Description.
        h (int): Description.
        m (int): Description.
        sec (float): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_jd2cal)
@icontract.require(lambda dj1: dj1 is not None, "dj1 cannot be None")
@icontract.require(lambda dj2: dj2 is not None, "dj2 cannot be None")
@icontract.ensure(lambda result: result is not None, "Jd2Cal output must not be None")
def jd2cal(dj1: float, dj2: float) -> tuple[int, int, int, float]:
    """Jd2cal.

    Args:
        dj1 (float): Description.
        dj2 (float): Description.

    Returns:
        tuple[int, int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_jd2calhms)
@icontract.require(lambda dj1: dj1 is not None, "dj1 cannot be None")
@icontract.require(lambda dj2: dj2 is not None, "dj2 cannot be None")
@icontract.ensure(lambda result: result is not None, "Jd2Calhms output must not be None")
def jd2calhms(dj1: float, dj2: float) -> tuple[int, int, int, int, int, float]:
    """Jd2calhms.

    Args:
        dj1 (float): Description.
        dj2 (float): Description.

    Returns:
        tuple[int, int, int, int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_utc2tai)
@icontract.require(lambda utc1: utc1 is not None, "utc1 cannot be None")
@icontract.require(lambda utc2: utc2 is not None, "utc2 cannot be None")
@icontract.ensure(lambda result: result is not None, "Utc2Tai output must not be None")
def utc2tai(utc1: float, utc2: float) -> float:
    """Utc2tai.

    Args:
        utc1 (float): Description.
        utc2 (float): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_tai2utc)
@icontract.require(lambda tai1: tai1 is not None, "tai1 cannot be None")
@icontract.require(lambda tai2: tai2 is not None, "tai2 cannot be None")
@icontract.ensure(lambda result: result is not None, "Tai2Utc output must not be None")
def tai2utc(tai1: float, tai2: float) -> float:
    """Tai2utc.

    Args:
        tai1 (float): Description.
        tai2 (float): Description.

    Returns:
        float: Description.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""


from juliacall import Main as jl


def _isleapyear_ffi(year):
    """Wrapper that calls the Julia version of isleapyear. Passes arguments through and returns the result."""
    return jl.eval("isleapyear(year)")

def _find_dayinyear_ffi(month, day, isleap):
    """Wrapper that calls the Julia version of find dayinyear. Passes arguments through and returns the result."""
    return jl.eval("find_dayinyear(month, day, isleap)")

def _find_year_ffi(d):
    """Wrapper that calls the Julia version of find year. Passes arguments through and returns the result."""
    return jl.eval("find_year(d)")

def _find_month_ffi(dayinyear, isleap):
    """Wrapper that calls the Julia version of find month. Passes arguments through and returns the result."""
    return jl.eval("find_month(dayinyear, isleap)")

def _find_day_ffi(dayinyear, month, isleap):
    """Wrapper that calls the Julia version of find day. Passes arguments through and returns the result."""
    return jl.eval("find_day(dayinyear, month, isleap)")

def _lastj2000dayofyear_ffi(year):
    """Wrapper that calls the Julia version of lastj2000 dayofyear. Passes arguments through and returns the result."""
    return jl.eval("lastj2000dayofyear(year)")

def _hms2fd_ffi(h, m, s):
    """Wrapper that calls the Julia version of hms2 fd. Passes arguments through and returns the result."""
    return jl.eval("hms2fd(h, m, s)")

def _fd2hms_ffi(fd):
    """Wrapper that calls the Julia version of fd2 hms. Passes arguments through and returns the result."""
    return jl.eval("fd2hms(fd)")

def _fd2hmsf_ffi(fd):
    """Wrapper that calls the Julia version of fd2 hmsf. Passes arguments through and returns the result."""
    return jl.eval("fd2hmsf(fd)")

def _cal2jd_ffi(Y, M, D):
    """Wrapper that calls the Julia version of cal2 jd. Passes arguments through and returns the result."""
    return jl.eval("cal2jd(Y, M, D)")

def _calhms2jd_ffi(Y, M, D, h, m, sec):
    """Wrapper that calls the Julia version of calhms2 jd. Passes arguments through and returns the result."""
    return jl.eval("calhms2jd(Y, M, D, h, m, sec)")

def _jd2cal_ffi(dj1, dj2):
    """Wrapper that calls the Julia version of jd2 cal. Passes arguments through and returns the result."""
    return jl.eval("jd2cal(dj1, dj2)")

def _jd2calhms_ffi(dj1, dj2):
    """Wrapper that calls the Julia version of jd2 calhms. Passes arguments through and returns the result."""
    return jl.eval("jd2calhms(dj1, dj2)")

def _utc2tai_ffi(utc1, utc2):
    """Wrapper that calls the Julia version of utc2 tai. Passes arguments through and returns the result."""
    return jl.eval("utc2tai(utc1, utc2)")

def _tai2utc_ffi(tai1, tai2):
    """Wrapper that calls the Julia version of tai2 utc. Passes arguments through and returns the result."""
    return jl.eval("tai2utc(tai1, tai2)")