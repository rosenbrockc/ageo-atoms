import numpy as np

from ageoa.e2e_ppg.kazemi_wrapper_d12.atoms import normalizesignal, wrapperevaluate


def test_normalizesignal_scales_to_unit_interval() -> None:
    result = normalizesignal(np.array([1.0, 3.0, 2.0], dtype=float))
    np.testing.assert_allclose(result, np.array([0.0, 1.0, 0.5], dtype=float))


def test_wrapperevaluate_matches_vendored_two_argument_surface() -> None:
    prediction = np.array([0.1] * 10 + [1.0] + [0.1] * 19, dtype=float)
    raw_signal = np.array([1.0] * 30, dtype=float)
    result = wrapperevaluate(prediction, raw_signal)
    np.testing.assert_array_equal(result, np.array([10], dtype=np.intp))
