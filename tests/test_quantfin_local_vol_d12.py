from __future__ import annotations

import icontract
import numpy as np
import pytest

from ageoa.quantfin.local_vol_d12.atoms import localvol, var


def test_var_computes_implied_variance() -> None:
    assert var(100.0, 1.0, 1.0, 0.2, {"surface": "flat"}) == 0.04000000000000001


def test_localvol_computes_dupire_ratio() -> None:
    result = localvol(0.18, 100.0, 0.2, {"curve": "flat"}, 100.0, 0.09, np.sqrt, 1.0, 0.04, 0.04)
    assert result == np.sqrt(2.0)


def test_var_rejects_non_positive_strike() -> None:
    with pytest.raises(icontract.ViolationError):
        var(0.0, 1.0, 1.0, 0.2, {"surface": "flat"})
