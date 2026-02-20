import numpy as np

from ageoa.e2e_ppg.atoms import process_ppg
from ageoa.e2e_ppg.state_models import PPGState


def test_process_ppg_empty_window_returns_empty_output():
    state = PPGState(sampling_rate=20, buffer=[1.0, 2.0, 3.0])

    out, new_state = process_ppg(np.array([]), state)
    assert out.shape == (0,)
    assert new_state.buffer == [1.0, 2.0, 3.0]


def test_process_ppg_output_shape_matches_input_window():
    state = PPGState(sampling_rate=20, buffer=[])
    samples = np.array([0.1, 0.2, 0.3])

    out, new_state = process_ppg(samples, state)
    assert out.shape == samples.shape
    assert len(new_state.buffer or []) >= len(samples)
