import numpy as np

from ageoa.biosppy.ecg_zz2018_d12.atoms import (
    assemblezz2018sqi,
    computebeatagreementsqi,
    computefrequencysqi,
    computekurtosissqi,
)


def test_computebeatagreementsqi_uses_upstream_defaults() -> None:
    detector_1 = np.array([100, 250, 400], dtype=int)
    detector_2 = np.array([100, 250, 400], dtype=int)
    assert computebeatagreementsqi(detector_1, detector_2) == 100.0


def test_frequency_and_kurtosis_sqi_use_defaults() -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 256, dtype=float))
    assert isinstance(computefrequencysqi(signal), float)
    assert isinstance(computekurtosissqi(signal), float)


def test_assemblezz2018sqi_returns_component_bundle() -> None:
    assert assemblezz2018sqi(0.9, 0.4, 3.2) == {"b_sqi": 0.9, "f_sqi": 0.4, "k_sqi": 3.2}
