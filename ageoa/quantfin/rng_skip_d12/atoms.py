from __future__ import annotations

"""Atom wrappers for the MWC64X random number generator with skip-ahead."""

import numpy as np
import icontract
from typing import Callable, Tuple

from ageoa.ghost.registry import register_atom
from .witnesses import (
    witness_addmod64,
    witness_f,
    witness_mulmod64,
    witness_next,
    witness_powmod64,
    witness_randomdouble,
    witness_randomint,
    witness_randomint64,
    witness_randomword32,
    witness_randomword64,
    witness_skip,
    witness_split,
)

import ctypes
import ctypes.util
from pathlib import Path


# ---------------------------------------------------------------------------
# randomword32
# ---------------------------------------------------------------------------

@register_atom(witness_randomword32)
@icontract.require(lambda state: isinstance(state, int) and state >= 0, "state must be a non-negative integer")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a (word32, new_state) tuple")
def randomword32(
    c: int,
    state: int,
    state_prime: int,
    x: int,
    xor: Callable,
) -> Tuple[int, int]:
    """Generate a 32-bit pseudo-random word from MWC64X state.

    Splits the 64-bit state into a carry *c* (upper 32 bits) and
    counter *x* (lower 32 bits), XORs them to produce the output
    word, then advances the state.

    Args:
        c: Upper 32 bits of state (carry).
        state: Current 64-bit generator state.
        state_prime: Updated 64-bit state after generation.
        x: Lower 32 bits of state (counter).
        xor: Bitwise XOR function.

    Returns:
        Tuple of (random 32-bit word, updated generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# randomint
# ---------------------------------------------------------------------------

@register_atom(witness_randomint)
@icontract.require(lambda g: isinstance(g, int) and g >= 0, "g -- generator state must be non-negative")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a (int, new_state) tuple")
def randomint(
    fromIntegral: Callable,
    g: int,
    g_prime: int,
    i: int,
) -> Tuple[int, int]:
    """Generate a random integer from Multiply With Carry 64-bit (MWC64X) state.

    Produces a 64-bit random word and converts it to a platform-width
    signed integer.

    Args:
        fromIntegral: Numeric type-conversion function.
        g: Current 64-bit generator state.
        g_prime: Updated state after generation.
        i: Converted integer result.

    Returns:
        Tuple of (random integer, updated generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# randomword64
# ---------------------------------------------------------------------------

@register_atom(witness_randomword64)
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a (word64, new_state) tuple")
def randomword64(
    buildWord64_prime: Callable,
    x: int,
    x_prime: int,
    y1: int,
    y2: int,
) -> Tuple[int, int]:
    """Generate a 64-bit pseudo-random word from MWC64X state.

    Calls randomWord32 twice and joins the two 32-bit halves into a
    single 64-bit word.

    Args:
        buildWord64_prime: Function to join two 32-bit words into 64 bits.
        x: Initial generator state.
        x_prime: Intermediate state after first 32-bit draw.
        y1: First 32-bit random word.
        y2: Second 32-bit random word.

    Returns:
        Tuple of (random 64-bit word, updated generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# randomdouble
# ---------------------------------------------------------------------------

@register_atom(witness_randomdouble)
@icontract.require(lambda x: isinstance(x, int) and x >= 0, "x -- generator state must be non-negative")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a (float, new_state) tuple")
def randomdouble(
    div: Callable,
    fromIntegral: Callable,
    val: int,
    x: int,
    x_prime: int,
) -> Tuple[float, int]:
    """Generate a uniform random double in [0, 1) from Multiply With Carry 64-bit (MWC64X) state.

    Draws a 64-bit word, drops the lowest 11 bits, and divides by
    2^53 to fill the full mantissa.

    Args:
        div: Integer division function.
        fromIntegral: Numeric type-conversion function.
        val: Intermediate 64-bit value after bit-shift.
        x: Current 64-bit generator state.
        x_prime: Updated state after generation.

    Returns:
        Tuple of (uniform double in [0, 1), updated generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# randomint64
# ---------------------------------------------------------------------------

@register_atom(witness_randomint64)
@icontract.require(lambda g: isinstance(g, int) and g >= 0, "g -- generator state must be non-negative")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a (int64, new_state) tuple")
def randomint64(
    fromIntegral: Callable,
    g: int,
    g_prime: int,
    i: int,
) -> Tuple[int, int]:
    """Generate a random 64-bit signed integer from Multiply With Carry 64-bit (MWC64X) state.

    Produces a 64-bit random word and reads it as a signed 64-bit value.

    Args:
        fromIntegral: Numeric type-conversion function.
        g: Current 64-bit generator state.
        g_prime: Updated state after generation.
        i: Converted 64-bit integer result.

    Returns:
        Tuple of (random signed 64-bit integer, updated generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# addmod64
# ---------------------------------------------------------------------------

@register_atom(witness_addmod64)
@icontract.require(lambda a: isinstance(a, int) and a >= 0, "a must be a non-negative integer")
@icontract.require(lambda b: isinstance(b, int) and b >= 0, "b must be a non-negative integer")
@icontract.require(lambda m: isinstance(m, int) and m > 0, "m -- modulus must be a positive integer")
@icontract.ensure(lambda result, m: isinstance(result, int) and 0 <= result < m, "result must be in [0, m)")
def addmod64(
    a: int,
    b: int,
    m: int,
    mod: Callable,
) -> int:
    """Compute (a + b) mod m for 64-bit unsigned integers.

    Args:
        a: First operand.
        b: Second operand.
        m: Modulus.
        mod: Modular arithmetic function.

    Returns:
        (a + b) mod m as a non-negative integer.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# mulmod64
# ---------------------------------------------------------------------------

@register_atom(witness_mulmod64)
@icontract.require(lambda a: isinstance(a, int) and a >= 0, "a must be a non-negative integer")
@icontract.require(lambda b: isinstance(b, int) and b >= 0, "b must be a non-negative integer")
@icontract.require(lambda m: isinstance(m, int) and m > 0, "m -- modulus must be a positive integer")
@icontract.ensure(lambda result, m: isinstance(result, int) and 0 <= result < m, "result must be in [0, m)")
def mulmod64(
    a: int,
    b: int,
    f: Callable,
    m: int,
) -> int:
    """Compute (a * b) mod m for 64-bit unsigned integers.

    Uses repeated doubling and conditional addition to avoid overflow
    on platforms without native 128-bit multiplication.

    Args:
        a: First operand.
        b: Second operand.
        f: Recursive helper that performs the shift-and-add loop.
        m: Modulus.

    Returns:
        (a * b) mod m as a non-negative integer.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# powmod64
# ---------------------------------------------------------------------------

@register_atom(witness_powmod64)
@icontract.require(lambda a: isinstance(a, int) and a >= 0, "a -- base must be non-negative")
@icontract.require(lambda e: isinstance(e, int) and e >= 0, "e -- exponent must be non-negative")
@icontract.require(lambda m: isinstance(m, int) and m > 0, "m -- modulus must be positive")
@icontract.ensure(lambda result, m: isinstance(result, int) and 0 <= result < m, "result must be in [0, m)")
def powmod64(
    a: int,
    e: int,
    f: Callable,
    m: int,
) -> int:
    """Compute a^e mod m for 64-bit unsigned integers.

    Uses binary exponentiation (square-and-multiply) with modular
    reduction at each step.

    Args:
        a: Base.
        e: Exponent.
        f: Recursive helper for the square-and-multiply loop.
        m: Modulus.

    Returns:
        a raised to the power e, modulo m.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# skip
# ---------------------------------------------------------------------------

@register_atom(witness_skip)
@icontract.require(lambda d: isinstance(d, int) and d >= 0, "d -- skip distance must be non-negative")
@icontract.require(lambda st: isinstance(st, int) and st >= 0, "st -- generator state must be non-negative")
@icontract.ensure(lambda result: isinstance(result, int) and result >= 0, "result must be a non-negative state")
def skip(
    d: int,
    st: int,
    st_prime: int,
) -> int:
    """Skip the MWC64X generator forward by *d* steps.

    Advances the generator state as if *d* random values had been
    drawn, using modular exponentiation for an O(log d) operation.

    Args:
        d: Number of steps to skip.
        st: Current 64-bit generator state.
        st_prime: Resulting state after the skip.

    Returns:
        Updated generator state advanced by *d* positions.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# next
# ---------------------------------------------------------------------------

@register_atom(witness_next)
@icontract.require(lambda g: isinstance(g, int) and g >= 0, "g -- generator state must be non-negative")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a (int, new_state) tuple")
def next(
    fromIntegral: Callable,
    g: int,
    g_prime: int,
    w: int,
) -> Tuple[int, int]:
    """Advance the generator by one step and return an integer.

    Implements the RandomGen next interface: draws a 64-bit word and
    converts it to a platform integer.

    Args:
        fromIntegral: Numeric type-conversion function.
        g: Current 64-bit generator state.
        g_prime: Updated state after generation.
        w: Raw 64-bit word produced.

    Returns:
        Tuple of (random integer, updated generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

@register_atom(witness_split)
@icontract.require(lambda g: isinstance(g, int) and g >= 0, "g -- generator state must be non-negative")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "result must be a two-element tuple of states")
def split(
    g: int,
    skip: Callable,
    skipConst: int,
) -> Tuple[int, int]:
    """Split the generator into two independent streams.

    One stream is advanced by the skip constant while the other
    retains the current state.

    Args:
        g: Current 64-bit generator state.
        skip: Skip-ahead function.
        skipConst: Fixed distance used to separate the two streams.

    Returns:
        Tuple of (advanced generator state, original generator state).
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# f  — mulmod64 inner loop
# ---------------------------------------------------------------------------

@register_atom(witness_f)
@icontract.ensure(lambda result: isinstance(result, int) and result >= 0, "result must be non-negative")
def f(
    a_prime: int,
    a1: int,
    b_prime: int,
    b1: int,
    f: Callable,
    otherwise: int,
    r: int,
    r_prime: int,
) -> int:
    """Execute one step of the shift-and-add loop for mulmod64.

    Checks the lowest bit of *a1*; if set, adds *b1* to the running
    result *r* modulo *m*.  Then doubles *b1* and halves *a1*.

    Args:
        a_prime: Shifted multiplicand (a >> 1).
        a1: Current multiplicand value.
        b_prime: Doubled multiplier (b + b mod m).
        b1: Current multiplier value.
        f: Recursive reference to this helper.
        otherwise: Result when lowest bit is not set.
        r: Running accumulator.
        r_prime: Updated accumulator after conditional add.

    Returns:
        Accumulated modular product after this step.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# f  — powmod64 inner loop
# ---------------------------------------------------------------------------

@register_atom(witness_f)
@icontract.ensure(lambda result: isinstance(result, int) and result >= 0, "result must be non-negative")
def f(
    acc: int,
    acc_prime: int,
    e_prime: int,
    e1: int,
    f: Callable,
    otherwise: int,
    sqr: int,
    sqr_prime: int,
) -> int:
    """Execute one step of the square-and-multiply loop for powmod64.

    Checks the lowest bit of *e1*; if set, multiplies *acc* by *sqr*
    modulo *m*.  Then squares *sqr* and halves *e1*.

    Args:
        acc: Running accumulator for the modular power.
        acc_prime: Updated accumulator after conditional multiply.
        e_prime: Shifted exponent (e >> 1).
        e1: Current exponent value.
        f: Recursive reference to this helper.
        otherwise: Accumulator value when lowest bit is not set.
        sqr: Current base being squared each iteration.
        sqr_prime: Squared base for the next iteration.

    Returns:
        Accumulated modular power after this step.
    """
    raise NotImplementedError("Wire to original implementation")


# ---------------------------------------------------------------------------
# FFI bindings (auto-generated, kept for reference)
# ---------------------------------------------------------------------------

def randomword32_ffi(c, state, state_prime, x, xor):
    """Wrapper that calls the Haskell version of randomword32."""
    _lib = ctypes.CDLL("./randomword32.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(c, state, state_prime, x, xor)

def randomint_ffi(fromIntegral, g, g_prime, i):
    """Wrapper that calls the Haskell version of randomint."""
    _lib = ctypes.CDLL("./randomint.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, g, g_prime, i)

def randomword64_ffi(buildWord64_prime, x, x_prime, y1, y2):
    """Wrapper that calls the Haskell version of randomword64."""
    _lib = ctypes.CDLL("./randomword64.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(buildWord64_prime, x, x_prime, y1, y2)

def randomdouble_ffi(div, fromIntegral, val, x, x_prime):
    """Wrapper that calls the Haskell version of randomdouble."""
    _lib = ctypes.CDLL("./randomdouble.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 5
    _func.restype = ctypes.c_void_p
    return _func(div, fromIntegral, val, x, x_prime)

def randomint64_ffi(fromIntegral, g, g_prime, i):
    """Wrapper that calls the Haskell version of randomint64."""
    _lib = ctypes.CDLL("./randomint64.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, g, g_prime, i)

def addmod64_ffi(a, b, m, mod):
    """Wrapper that calls the Haskell version of addmod64."""
    _lib = ctypes.CDLL("./addmod64.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(a, b, m, mod)

def mulmod64_ffi(a, b, f, m):
    """Wrapper that calls the Haskell version of mulmod64."""
    _lib = ctypes.CDLL("./mulmod64.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(a, b, f, m)

def powmod64_ffi(a, e, f, m):
    """Wrapper that calls the Haskell version of powmod64."""
    _lib = ctypes.CDLL("./powmod64.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(a, e, f, m)

def skip_ffi(d, st, st_prime):
    """Wrapper that calls the Haskell version of skip."""
    _lib = ctypes.CDLL("./skip.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 3
    _func.restype = ctypes.c_void_p
    return _func(d, st, st_prime)

def next_ffi(fromIntegral, g, g_prime, w):
    """Wrapper that calls the Haskell version of next."""
    _lib = ctypes.CDLL("./next.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 4
    _func.restype = ctypes.c_void_p
    return _func(fromIntegral, g, g_prime, w)

def split_ffi(g, skip, skipConst):
    """Wrapper that calls the Haskell version of split."""
    _lib = ctypes.CDLL("./split.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 3
    _func.restype = ctypes.c_void_p
    return _func(g, skip, skipConst)

def f_mulmod_ffi(a_prime, a1, b_prime, b1, f, otherwise, r, r_prime):
    """Wrapper that calls the Haskell version of f (mulmod64 loop)."""
    _lib = ctypes.CDLL("./f.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 8
    _func.restype = ctypes.c_void_p
    return _func(a_prime, a1, b_prime, b1, f, otherwise, r, r_prime)

def f_powmod_ffi(acc, acc_prime, e_prime, e1, f, otherwise, sqr, sqr_prime):
    """Wrapper that calls the Haskell version of f (powmod64 loop)."""
    _lib = ctypes.CDLL("./f.so")
    _func_name = 'placeholder'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p] * 8
    _func.restype = ctypes.c_void_p
    return _func(acc, acc_prime, e_prime, e1, f, otherwise, sqr, sqr_prime)
