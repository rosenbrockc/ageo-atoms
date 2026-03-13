from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal
def witness_isleapyear(year: AbstractArray, *args, **kwargs):
    """Ghost witness for Isleapyear."""
    result = AbstractArray(
        shape=year.shape,
        dtype="float64",
    )
    return result

def witness_find_dayinyear(month: AbstractArray, day: AbstractArray, isleap: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Dayinyear."""
    result = AbstractArray(
        shape=month.shape,
        dtype="float64",
    )
    return result

def witness_find_year(d: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Year."""
    result = AbstractArray(
        shape=d.shape,
        dtype="float64",
    )
    return result

def witness_find_month(dayinyear: AbstractArray, isleap: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Month."""
    result = AbstractArray(
        shape=dayinyear.shape,
        dtype="float64",
    )
    return result

def witness_find_day(dayinyear: AbstractArray, month: AbstractArray, isleap: AbstractArray) -> AbstractArray:
    """Ghost witness for Find Day."""
    result = AbstractArray(
        shape=dayinyear.shape,
        dtype="float64",
    )
    return result

def witness_lastj2000dayofyear(year: AbstractArray) -> AbstractArray:
    """Ghost witness for Lastj2000Dayofyear."""
    result = AbstractArray(
        shape=year.shape,
        dtype="float64",
    )
    return result

def witness_hms2fd(h: AbstractArray, m: AbstractArray, s: AbstractArray) -> AbstractArray:
    """Ghost witness for Hms2Fd."""
    result = AbstractArray(
        shape=h.shape,
        dtype="float64",
    )
    return result

def witness_fd2hms(fd: AbstractArray) -> AbstractArray:
    """Ghost witness for Fd2Hms."""
    result = AbstractArray(
        shape=fd.shape,
        dtype="float64",
    )
    return result

def witness_fd2hmsf(fd: AbstractArray) -> AbstractArray:
    """Ghost witness for Fd2Hmsf."""
    result = AbstractArray(
        shape=fd.shape,
        dtype="float64",
    )
    return result

def witness_cal2jd(Y: AbstractArray, M: AbstractArray, D: AbstractArray) -> AbstractArray:
    """Ghost witness for Cal2Jd."""
    result = AbstractArray(
        shape=Y.shape,
        dtype="float64",
    )
    return result

def witness_calhms2jd(Y: AbstractArray, M: AbstractArray, D: AbstractArray, h: AbstractArray, m: AbstractArray, sec: AbstractArray) -> AbstractArray:
    """Ghost witness for Calhms2Jd."""
    result = AbstractArray(
        shape=Y.shape,
        dtype="float64",
    )
    return result

def witness_jd2cal(dj1: AbstractArray, dj2: AbstractArray) -> AbstractArray:
    """Ghost witness for Jd2Cal."""
    result = AbstractArray(
        shape=dj1.shape,
        dtype="float64",
    )
    return result

def witness_jd2calhms(dj1: AbstractArray, dj2: AbstractArray) -> AbstractArray:
    """Ghost witness for Jd2Calhms."""
    result = AbstractArray(
        shape=dj1.shape,
        dtype="float64",
    )
    return result

def witness_utc2tai(utc1: AbstractArray, utc2: AbstractArray) -> AbstractArray:
    """Ghost witness for Utc2Tai."""
    result = AbstractArray(
        shape=utc1.shape,
        dtype="float64",
    )
    return result

def witness_tai2utc(tai1: AbstractArray, tai2: AbstractArray) -> AbstractArray:
    """Ghost witness for Tai2Utc."""
    result = AbstractArray(
        shape=tai1.shape,
        dtype="float64",
    )
    return result