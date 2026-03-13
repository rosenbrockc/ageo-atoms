from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_randomword32(c: AbstractArray, state: AbstractArray, state_prime: AbstractArray, x: AbstractArray, xor: AbstractArray) -> AbstractArray:
    """Ghost witness for Randomword32."""
    result = AbstractArray(
        shape=c.shape,
        dtype="float64",
    )
    return result

def witness_randomint(fromIntegral: AbstractArray, g: AbstractArray, g_prime: AbstractArray, i: AbstractArray) -> AbstractArray:
    """Ghost witness for Randomint."""
    result = AbstractArray(
        shape=fromIntegral.shape,
        dtype="float64",
    )
    return result

def witness_randomword64(buildWord64_prime: AbstractArray, x: AbstractArray, x_prime: AbstractArray, y1: AbstractArray, y2: AbstractArray) -> AbstractArray:
    """Ghost witness for Randomword64."""
    result = AbstractArray(
        shape=buildWord64_prime.shape,
        dtype="float64",
    )
    return result

def witness_randomdouble(div: AbstractArray, fromIntegral: AbstractArray, val: AbstractArray, x: AbstractArray, x_prime: AbstractArray) -> AbstractArray:
    """Ghost witness for Randomdouble."""
    result = AbstractArray(
        shape=div.shape,
        dtype="float64",
    )
    return result

def witness_randomint64(fromIntegral: AbstractArray, g: AbstractArray, g_prime: AbstractArray, i: AbstractArray) -> AbstractArray:
    """Ghost witness for Randomint64."""
    result = AbstractArray(
        shape=fromIntegral.shape,
        dtype="float64",
    )
    return result

def witness_addmod64(a: AbstractArray, b: AbstractArray, m: AbstractArray, mod: AbstractArray) -> AbstractArray:
    """Ghost witness for Addmod64."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result

def witness_mulmod64(a: AbstractArray, b: AbstractArray, f: AbstractArray, m: AbstractArray) -> AbstractArray:
    """Ghost witness for Mulmod64."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result

def witness_powmod64(a: AbstractArray, e: AbstractArray, f: AbstractArray, m: AbstractArray) -> AbstractArray:
    """Ghost witness for Powmod64."""
    result = AbstractArray(
        shape=a.shape,
        dtype="float64",
    )
    return result

def witness_skip(d: AbstractArray, st: AbstractArray, st_prime: AbstractArray) -> AbstractArray:
    """Ghost witness for Skip."""
    result = AbstractArray(
        shape=d.shape,
        dtype="float64",
    )
    return result

def witness_next(fromIntegral: AbstractArray, g: AbstractArray, g_prime: AbstractArray, w: AbstractArray) -> AbstractArray:
    """Ghost witness for Next."""
    result = AbstractArray(
        shape=fromIntegral.shape,
        dtype="float64",
    )
    return result

def witness_split(g: AbstractArray, skip: AbstractArray, skipConst: AbstractArray) -> AbstractArray:
    """Ghost witness for Split."""
    result = AbstractArray(
        shape=g.shape,
        dtype="float64",
    )
    return result

def witness_f_prime(a_prime: AbstractArray, a1: AbstractArray, b_prime: AbstractArray, b1: AbstractArray, f: AbstractArray, otherwise: AbstractArray, r: AbstractArray, r_prime: AbstractArray) -> AbstractArray:
    """Ghost witness for F."""
    result = AbstractArray(
        shape=a_prime.shape,
        dtype="float64",
    )
    return result

def witness_f(acc: AbstractArray, acc_prime: AbstractArray, e_prime: AbstractArray, e1: AbstractArray, f: AbstractArray, otherwise: AbstractArray, sqr: AbstractArray, sqr_prime: AbstractArray) -> AbstractArray:
    """Ghost witness for F."""
    result = AbstractArray(
        shape=acc.shape,
        dtype="float64",
    )
    return result
