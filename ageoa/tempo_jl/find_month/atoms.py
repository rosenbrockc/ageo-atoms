from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_date, witness_datetime, witness_show, witness_time

from juliacall import Main as jl


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_date)
@icontract.require(lambda offset: offset is not None, "offset cannot be None")
@icontract.ensure(lambda result: result is not None, "Date output must not be None")
def date(offset: int) -> tuple[int, int, int]:
    """Date.

    Args:
        offset (int): Description.

    Returns:
        tuple[int, int, int]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_date)
@icontract.require(lambda year: year is not None, "year cannot be None")
@icontract.require(lambda dayinyear: dayinyear is not None, "dayinyear cannot be None")
@icontract.ensure(lambda result: result is not None, "Date output must not be None")
def date(year: int, dayinyear: int) -> tuple[int, int, int]:
    """Date.

    Args:
        year (int): Description.
        dayinyear (int): Description.

    Returns:
        tuple[int, int, int]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda d: d is not None, "d cannot be None")
@icontract.ensure(lambda result: result is not None, "Show output must not be None")
def show(io: str, d: str) -> str:
    """Show.

    Args:
        io (str): Description.
        d (str): Description.

    Returns:
        str: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_time)
@icontract.require(lambda hour: hour is not None, "hour cannot be None")
@icontract.require(lambda minute: minute is not None, "minute cannot be None")
@icontract.require(lambda second: second is not None, "second cannot be None")
@icontract.ensure(lambda result: result is not None, "Time output must not be None")
def time(hour: int, minute: int, second: float) -> tuple[int, int, float]:
    """Time.

    Args:
        hour (int): Description.
        minute (int): Description.
        second (float): Description.

    Returns:
        tuple[int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_time)
@icontract.require(lambda secondinday: secondinday is not None, "secondinday cannot be None")
@icontract.require(lambda fraction: fraction is not None, "fraction cannot be None")
@icontract.ensure(lambda result: result is not None, "Time output must not be None")
def time(secondinday: int, fraction: float) -> tuple[int, int, float]:
    """Time.

    Args:
        secondinday (int): Description.
        fraction (float): Description.

    Returns:
        tuple[int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_time)
@icontract.require(lambda secondinday: secondinday is not None, "secondinday cannot be None")
@icontract.ensure(lambda result: result is not None, "Time output must not be None")
def time(secondinday: int) -> tuple[int, int, float]:
    """Time.

    Args:
        secondinday (int): Description.

    Returns:
        tuple[int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_show)
@icontract.require(lambda io: io is not None, "io cannot be None")
@icontract.require(lambda t: t is not None, "t cannot be None")
@icontract.ensure(lambda result: result is not None, "Show output must not be None")
def show(io: str, t: str) -> str:
    """Show.

    Args:
        io (str): Description.
        t (str): Description.

    Returns:
        str: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_datetime)
@icontract.require(lambda year, month, day: 1 <= month <= 12 and 1 <= day <= 31, "month must be 1-12, day must be 1-31")
@icontract.ensure(lambda result: result is not None, "Datetime output must not be None")
def datetime(year: int, month: int, day: int, hour: int, min: int, sec: float) -> tuple[int, int, int, int, int, float]:
    """Datetime.

    Args:
        year (int): Description.
        month (int): Description.
        day (int): Description.
        hour (int): Description.
        min (int): Description.
        sec (float): Description.

    Returns:
        tuple[int, int, int, int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_datetime)
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.ensure(lambda result: result is not None, "Datetime output must not be None")
def datetime(s: str) -> tuple[int, int, int, int, int, float]:
    """Datetime.

    Args:
        s (str): Description.

    Returns:
        tuple[int, int, int, int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_datetime)
@icontract.require(lambda seconds: seconds is not None, "seconds cannot be None")
@icontract.ensure(lambda result: result is not None, "Datetime output must not be None")
def datetime(seconds: float) -> tuple[int, int, int, int, int, float]:
    """Datetime.

    Args:
        seconds (float): Description.

    Returns:
        tuple[int, int, int, int, int, float]: Description.
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""


from juliacall import Main as jl


def _date_ffi(offset):
    """FFI bridge to Julia implementation of Date."""
    return jl.eval("date(offset)")

def _date_ffi(year, dayinyear):
    """FFI bridge to Julia implementation of Date."""
    return jl.eval("date(year, dayinyear)")

def _show_ffi(io, d):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, d)")

def _time_ffi(hour, minute, second):
    """FFI bridge to Julia implementation of Time."""
    return jl.eval("time(hour, minute, second)")

def _time_ffi(secondinday, fraction):
    """FFI bridge to Julia implementation of Time."""
    return jl.eval("time(secondinday, fraction)")

def _time_ffi(secondinday):
    """FFI bridge to Julia implementation of Time."""
    return jl.eval("time(secondinday)")

def _show_ffi(io, t):
    """FFI bridge to Julia implementation of Show."""
    return jl.eval("show(io, t)")

def _datetime_ffi(year, month, day, hour, min, sec):
    """FFI bridge to Julia implementation of Datetime."""
    return jl.eval("datetime(year, month, day, hour, min, sec)")

def _datetime_ffi(s):
    """FFI bridge to Julia implementation of Datetime."""
    return jl.eval("datetime(s)")

def _datetime_ffi(seconds):
    """FFI bridge to Julia implementation of Datetime."""
    return jl.eval("datetime(seconds)")