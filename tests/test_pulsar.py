import numpy as np
import pytest

from ageoa.pulsar.pipeline import SNR, de_disperse, delay_from_DM, fold_signal


def test_pulsar_pipeline():
    fchan = 1400.0
    width = 0.5
    tsamp = 0.00035
    period = 100
    dm = 10.0

    # 1. Generate synthetic spectrogram (noise + one dispersed pulse)
    n_time = 1000
    n_chan = 32
    data = np.random.rand(n_time, n_chan) * 0.1

    # Add a pulse dispersed by DM=10.0
    for i in range(n_chan):
        freq = i * width + fchan
        delay = int(delay_from_DM(dm, freq) / tsamp)
        idx = (500 + delay) % n_time
        data[idx, i] += 1.0

    # 2. Dedisperse
    dedispersed = de_disperse(data, dm, fchan, width, tsamp)
    assert dedispersed.shape == data.shape

    # 3. Fold
    profile = fold_signal(dedispersed, period)
    assert profile.shape == (period,)

    # 4. SNR
    snr_val = SNR(profile)
    assert snr_val > 0


def test_delay_logic():
    # Delay should decrease with frequency
    d1 = delay_from_DM(10.0, 1000.0)
    d2 = delay_from_DM(10.0, 2000.0)
    assert d1 > d2
    assert delay_from_DM(0.0, 1000.0) == 0.0


def test_fold_signal_uses_all_period_blocks():
    period = 4
    # Two full periods: first all zeros, second all ones.
    data = np.vstack([np.zeros((period, 1)), np.ones((period, 1))])

    profile = fold_signal(data, period)
    # Averaging both blocks should yield 0.5 across profile bins.
    assert np.allclose(profile, 0.5)


def test_dedisperse_requires_positive_tsamp():
    with pytest.raises(Exception):
        de_disperse(np.zeros((10, 2)), DM=10.0, fchan=1400.0, width=0.5, tsamp=0.0)
