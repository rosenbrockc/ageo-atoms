from __future__ import annotations

import icontract
import numpy as np
import pytest

from ageoa.quantfin.char_func_option_d12.atoms import cf, charfuncoption


def test_cf_evaluates_martingale_corrected_characteristic_function() -> None:
    result = cf(
        lambda model, fg, tmat: lambda x: complex(model["scale"] * x.real + fg["offset"], x.imag + tmat),
        {"offset": 0.25},
        {"scale": 2.0},
        1.5,
        complex(0.5, -0.25),
    )
    assert result == complex(1.25, 1.25)


def test_charfuncoption_prices_deterministic_zero_integral_case() -> None:
    result = charfuncoption(
        0.0,
        lambda x: x,
        lambda model, fg, tmat: lambda x: complex(x.real + model["shift"], x.imag + fg["shift"] + tmat),
        0.95,
        0.1,
        0.2,
        lambda t: 0.95,
        np.exp,
        lambda exp_fn, imag, k, left, real_part, right, v, v_prime: float(real_part(exp_fn(-imag * v * k) * left * right)),
        {"shift": 0.0},
        lambda v: 0.0,
        lambda v: 0.0,
        1j,
        lambda func_inner, lower, upper, tol: 0.0,
        float(np.log(100.0)),
        complex(2.0, 0.0),
        np.log,
        {"shift": 0.0},
        "call",
        0.0,
        0.0,
        float(np.pi),
        1.0,
        lambda z: float(z.real),
        complex(3.0, 0.0),
        100.0,
        100.0,
        1.0,
        complex(0.0, 0.0),
        complex(0.0, 0.0),
        0.0,
        {"curve": "flat"},
    )
    assert result == 0.0


def test_charfuncoption_rejects_non_positive_strike() -> None:
    with pytest.raises(icontract.ViolationError):
        charfuncoption(
            0.0,
            lambda x: x,
            lambda model, fg, tmat: lambda x: complex(x.real, x.imag),
            0.95,
            0.1,
            0.2,
            lambda t: 0.95,
            np.exp,
            lambda exp_fn, imag, k, left, real_part, right, v, v_prime: 0.0,
            {"shift": 0.0},
            lambda v: 0.0,
            lambda v: 0.0,
            1j,
            lambda func_inner, lower, upper, tol: 0.0,
            float(np.log(100.0)),
            complex(2.0, 0.0),
            np.log,
            {"shift": 0.0},
            "call",
            0.0,
            0.0,
            float(np.pi),
            1.0,
            lambda z: float(z.real),
            complex(3.0, 0.0),
            100.0,
            0.0,
            1.0,
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            0.0,
            {"curve": "flat"},
        )
