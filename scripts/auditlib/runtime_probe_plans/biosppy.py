"""BioSPPy family runtime probe plans split from the monolithic registry."""

from __future__ import annotations

from typing import Any, Callable

import importlib
import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_dict_keys = rt._assert_dict_keys
    _assert_monotonic_index_array = rt._assert_monotonic_index_array
    _assert_online_filter_init_state = rt._assert_online_filter_init_state
    _assert_online_filter_step_result = rt._assert_online_filter_step_result
    _assert_pair_of_arrays = rt._assert_pair_of_arrays
    _assert_pair_of_sorted_integer_arrays = rt._assert_pair_of_sorted_integer_arrays
    _assert_scalar = rt._assert_scalar
    _assert_shape = rt._assert_shape
    _assert_sorted_array = rt._assert_sorted_array
    _assert_triple_of_arrays_matching_onsets = rt._assert_triple_of_arrays_matching_onsets
    _assert_value = rt._assert_value

    def _biosppy_detector_plans() -> dict[str, ProbePlan]:
        def _synthetic_ecg() -> np.ndarray:
            fs = 1000.0
            duration = 10.0
            heart_rate = 75.0
            t = np.linspace(0.0, duration, int(duration * fs), endpoint=False)
            f0 = heart_rate / 60.0
            signal = np.zeros_like(t)
            peak_times = np.arange(0.5, duration, 1.0 / f0)
            for peak in peak_times:
                signal += 1.5 * np.exp(-((t - peak) ** 2) / (2 * (0.005 ** 2)))
            rng = np.random.RandomState(7)
            signal += 0.05 * rng.normal(size=len(t))
            return signal

        def _assert_peak_indices(result: Any) -> None:
            peaks = np.asarray(result)
            assert peaks.ndim == 1
            assert peaks.size > 0
            assert np.all(np.diff(peaks) >= 0)
            assert np.all(peaks >= 0)
            assert np.all(peaks < 10_000)

        signal = _synthetic_ecg()
        ppg_sampling_rate = 100.0
        ppg_time = np.linspace(0.0, 10.0, int(10.0 * ppg_sampling_rate), endpoint=False)
        ppg_signal = np.full_like(ppg_time, 0.02)
        for center in np.arange(0.5, 10.0, 1.0):
            ppg_signal += np.exp(-((ppg_time - center) ** 2) / (2 * (0.03 ** 2)))

        emg_sampling_rate = 1000.0
        emg_time = np.linspace(0.0, 2.0, int(2.0 * emg_sampling_rate), endpoint=False)
        emg_rest = 0.01 * np.sin(2 * np.pi * 10 * np.linspace(0.0, 0.4, int(0.4 * emg_sampling_rate), endpoint=False))
        emg_signal = 0.01 * np.sin(2 * np.pi * 10 * emg_time)
        emg_signal[700:1100] += 0.5 * np.sin(np.linspace(0.0, np.pi, 400))
        emg_signal[1300:1600] += 0.7 * np.sin(np.linspace(0.0, np.pi, 300))

        eda_sampling_rate = 100.0
        eda_time = np.linspace(0.0, 20.0, int(20.0 * eda_sampling_rate), endpoint=False)
        eda_signal = 0.1 + 0.02 * np.sin(2 * np.pi * 0.1 * eda_time)
        for center in (4.0, 10.0, 16.0):
            eda_signal += 0.5 * np.exp(-np.maximum(eda_time - center, 0.0) / 1.2) * (eda_time >= center)

        pcg_sampling_rate = 1000.0
        pcg_time = np.linspace(0.0, 4.0, int(4.0 * pcg_sampling_rate), endpoint=False)
        pcg_signal = np.zeros_like(pcg_time)
        for s1, s2 in [(0.4, 0.7), (1.4, 1.7), (2.4, 2.7), (3.4, 3.7)]:
            pcg_signal += np.exp(-((pcg_time - s1) ** 2) / (2 * (0.01 ** 2)))
            pcg_signal += 0.7 * np.exp(-((pcg_time - s2) ** 2) / (2 * (0.012 ** 2)))

        abp_sampling_rate = 1000.0
        abp_time = np.linspace(0.0, 10.0, int(10.0 * abp_sampling_rate), endpoint=False)
        abp_signal = np.zeros_like(abp_time)
        for center in np.arange(0.5, 10.0, 1.0):
            abp_signal += 0.8 * np.exp(-((abp_time - center) ** 2) / (2 * (0.02 ** 2)))
            abp_signal += 0.3 * np.exp(-((abp_time - (center + 0.08)) ** 2) / (2 * (0.03 ** 2)))

        return {
            "ageoa.biosppy.abp.audio_onset_detection": ProbePlan(
                positive=ProbeCase(
                    "ABP onset detection returns monotonic onset indices on a synthetic pulse trace",
                    lambda func: func(abp_signal, abp_sampling_rate),
                    _assert_monotonic_index_array(max_value=len(abp_signal) - 1),
                ),
                negative=ProbeCase(
                    "ABP onset detection rejects a missing signal",
                    lambda func: func(None, abp_sampling_rate),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.bandpass_filter": ProbePlan(
                positive=ProbeCase(
                    "ECG bandpass filtering preserves waveform shape on a synthetic ECG trace",
                    lambda func: func(signal, sampling_rate=1000.0),
                    _assert_shape(signal.shape),
                ),
                negative=ProbeCase(
                    "ECG bandpass filtering rejects a missing signal",
                    lambda func: func(None, sampling_rate=1000.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.r_peak_detection": ProbePlan(
                positive=ProbeCase(
                    "R-peak detection returns monotonic peak indices on a synthetic ECG trace",
                    lambda func: func(signal, sampling_rate=1000.0),
                    _assert_monotonic_index_array(max_value=len(signal) - 1),
                ),
                negative=ProbeCase(
                    "R-peak detection rejects a negative sampling rate",
                    lambda func: func(signal, sampling_rate=-1.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.peak_correction": ProbePlan(
                positive=ProbeCase(
                    "Peak correction returns monotonic corrected peak indices on a synthetic ECG trace",
                    lambda func: func(signal, np.array([500, 1300, 2100, 2900, 3700, 4500, 5300, 6100, 6900, 7700], dtype=int), sampling_rate=1000.0),
                    _assert_monotonic_index_array(max_value=len(signal) - 1),
                ),
                negative=ProbeCase(
                    "Peak correction rejects a missing filtered signal",
                    lambda func: func(None, np.array([500, 1300], dtype=int), sampling_rate=1000.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.template_extraction": ProbePlan(
                positive=ProbeCase(
                    "Template extraction returns templates and aligned peaks on a synthetic ECG trace",
                    lambda func: func(signal, np.array([500, 1300, 2100, 2900, 3700, 4500, 5300, 6100, 6900, 7700], dtype=int), sampling_rate=1000.0),
                    _assert_pair_of_arrays(),
                ),
                negative=ProbeCase(
                    "Template extraction rejects a missing filtered signal",
                    lambda func: func(None, np.array([500, 1300], dtype=int), sampling_rate=1000.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.heart_rate_computation": ProbePlan(
                positive=ProbeCase(
                    "Heart-rate computation returns aligned index and bpm arrays",
                    lambda func: func(np.array([500, 1300, 2100, 2900, 3700, 4500, 5300, 6100, 6900, 7700], dtype=int), sampling_rate=1000.0),
                    _assert_pair_of_arrays(),
                ),
                negative=ProbeCase(
                    "Heart-rate computation rejects a negative sampling rate",
                    lambda func: func(np.array([500, 1300, 2100], dtype=int), sampling_rate=-1.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.ssf_segmenter": ProbePlan(
                positive=ProbeCase(
                    "SSF segmenter returns monotonic peak indices on a synthetic ECG trace",
                    lambda func: func(signal, sampling_rate=1000.0),
                    _assert_monotonic_index_array(max_value=len(signal) - 1),
                ),
                negative=ProbeCase(
                    "SSF segmenter rejects a missing signal",
                    lambda func: func(None, sampling_rate=1000.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg.christov_segmenter": ProbePlan(
                positive=ProbeCase(
                    "Christov segmenter returns monotonic peak indices on a synthetic ECG trace",
                    lambda func: func(signal, sampling_rate=1000.0),
                    _assert_monotonic_index_array(max_value=len(signal) - 1),
                ),
                negative=ProbeCase(
                    "Christov segmenter rejects a missing signal",
                    lambda func: func(None, sampling_rate=1000.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_detectors.hamilton_segmentation": ProbePlan(
                positive=ProbeCase(
                    "Hamilton ECG segmentation detects peaks on a synthetic ECG trace",
                    lambda func: func(signal, 1000.0),
                    _assert_peak_indices,
                ),
                negative=ProbeCase(
                    "Hamilton ECG segmentation rejects a missing signal",
                    lambda func: func(None, 1000.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_detectors.thresholdbasedsignalsegmentation": ProbePlan(
                positive=ProbeCase(
                    "ASI threshold segmentation detects peaks on a synthetic ECG trace",
                    lambda func: func(signal, 1000.0, 5.0),
                    _assert_peak_indices,
                ),
                negative=ProbeCase(
                    "ASI threshold segmentation rejects a missing signal",
                    lambda func: func(None, 1000.0, 5.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_detectors.hamilton_segmenter": ProbePlan(
                positive=ProbeCase(
                    "Hamilton ECG segmenter detects peaks on a synthetic ECG trace",
                    lambda func: func(signal, 1000.0),
                    _assert_peak_indices,
                ),
                negative=ProbeCase(
                    "Hamilton ECG segmenter rejects a non-numeric sampling rate",
                    lambda func: func(signal, "bad"),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.eda.gamboa_segmenter": ProbePlan(
                positive=ProbeCase(
                    "EDA onset segmentation returns monotonic indices on a synthetic phasic signal",
                    lambda func: func(eda_signal, eda_sampling_rate),
                    _assert_monotonic_index_array(max_value=len(eda_signal) - 1),
                ),
                negative=ProbeCase(
                    "EDA onset segmentation rejects an empty signal",
                    lambda func: func(np.asarray([], dtype=float), eda_sampling_rate),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.eda.eda_feature_extraction": ProbePlan(
                positive=ProbeCase(
                    "EDA feature extraction returns aligned amplitude, rise-time, and decay arrays",
                    lambda func: func(eda_signal, np.array([400, 1000, 1600], dtype=int), eda_sampling_rate),
                    _assert_triple_of_arrays_matching_onsets(),
                ),
                negative=ProbeCase(
                    "EDA feature extraction rejects a missing signal",
                    lambda func: func(None, np.array([400, 1000], dtype=int), eda_sampling_rate),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ppg_detectors.detect_signal_onsets_elgendi2013": ProbePlan(
                positive=ProbeCase(
                    "Elgendi PPG onset detection finds the synthetic pulse train onsets",
                    lambda func: func(ppg_signal, ppg_sampling_rate, 0.111, 0.667, 0.02, 0.3),
                    _assert_sorted_array(np.array([50, 150, 250, 350, 450, 550, 650, 750, 850, 950])),
                ),
                negative=ProbeCase(
                    "Elgendi PPG onset detection rejects a non-numeric sampling rate",
                    lambda func: func(ppg_signal, "bad", 0.111, 0.667, 0.02, 0.3),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ppg_detectors.detectonsetevents": ProbePlan(
                positive=ProbeCase(
                    "Kavsaoğlu PPG onset detection finds the synthetic pulse train events",
                    lambda func: func(ppg_signal, ppg_sampling_rate, 0.2, 4, 60.0, 0.3, 180.0),
                    _assert_sorted_array(np.array([78, 178, 278, 378, 478, 578, 678, 778, 878])),
                ),
                negative=ProbeCase(
                    "Kavsaoğlu PPG onset detection rejects a missing signal",
                    lambda func: func(None, ppg_sampling_rate, 0.2, 4, 60.0, 0.3, 180.0),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.pcg.shannon_energy": ProbePlan(
                positive=ProbeCase(
                    "PCG Shannon-energy envelope preserves signal shape and non-negativity",
                    lambda func: func(pcg_signal),
                    _assert_shape(pcg_signal.shape),
                ),
                negative=ProbeCase(
                    "PCG Shannon-energy envelope rejects a non-array signal",
                    lambda func: func(None),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.pcg.pcg_segmentation": ProbePlan(
                positive=ProbeCase(
                    "PCG segmentation returns alternating S1 and S2 peaks from a synthetic envelope",
                    lambda func: func(np.maximum(pcg_signal, 0.0), pcg_sampling_rate),
                    _assert_pair_of_sorted_integer_arrays(),
                ),
                negative=ProbeCase(
                    "PCG segmentation rejects an empty envelope",
                    lambda func: func(np.asarray([], dtype=float), pcg_sampling_rate),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.emg_detectors.detect_onsets_with_rest_aware_thresholds": ProbePlan(
                positive=ProbeCase(
                    "rest-aware EMG onset detection returns a valid empty onset array for the quiet synthetic trace",
                    lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 20, 10, 1.0, 0.5),
                    _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
                ),
                negative=ProbeCase(
                    "rest-aware EMG onset detection rejects a missing signal",
                    lambda func: func(None, emg_rest, emg_sampling_rate, 20, 10, 1.0, 0.5),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.emg_detectors.bonato_onset_detection": ProbePlan(
                positive=ProbeCase(
                    "Bonato EMG onset detection returns a valid onset array for the quiet synthetic trace",
                    lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 1.0, 0.05, 3, 2),
                    _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
                ),
                negative=ProbeCase(
                    "Bonato EMG onset detection rejects a missing signal",
                    lambda func: func(None, emg_rest, emg_sampling_rate, 1.0, 0.05, 3, 2),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.emg_detectors.threshold_based_onset_detection": ProbePlan(
                positive=ProbeCase(
                    "threshold-based EMG onset detection returns a valid onset array for the quiet synthetic trace",
                    lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 1.0, 0.05),
                    _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
                ),
                negative=ProbeCase(
                    "threshold-based EMG onset detection rejects a missing signal",
                    lambda func: func(None, emg_rest, emg_sampling_rate, 1.0, 0.05),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.emg_detectors.solnik_onset_detect": ProbePlan(
                positive=ProbeCase(
                    "Solnik EMG onset detection returns a valid onset array for the quiet synthetic trace",
                    lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 1.0, 0.05),
                    _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
                ),
                negative=ProbeCase(
                    "Solnik EMG onset detection rejects a missing signal",
                    lambda func: func(None, emg_rest, emg_sampling_rate, 1.0, 0.05),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }


    def _biosppy_sqi_plans() -> dict[str, ProbePlan]:
        signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
        detector_1 = np.array([20, 60, 100, 140, 180])
        detector_2 = np.array([21, 59, 101, 141, 179])
        return {
            "ageoa.biosppy.ecg_zz2018.calculatecompositesqi_zz2018": ProbePlan(
                positive=ProbeCase(
                    "ZZ2018 composite SQI classifies a small synthetic signal",
                    lambda func: func(signal, detector_1, detector_2, 1000.0, 50, 64, "simple"),
                    _assert_value("Barely acceptable"),
                ),
                negative=ProbeCase(
                    "ZZ2018 composite SQI rejects a non-numeric sampling rate",
                    lambda func: func(signal, detector_1, detector_2, "bad", 50, 64, "simple"),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_zz2018.calculatebeatagreementsqi": ProbePlan(
                positive=ProbeCase(
                    "beat-agreement SQI returns the expected agreement score",
                    lambda func: func(detector_1, detector_2, 1000.0, "simple", 50),
                    _assert_scalar(100.0),
                ),
                negative=ProbeCase(
                    "beat-agreement SQI rejects a non-numeric sampling rate",
                    lambda func: func(detector_1, detector_2, "bad", "simple", 50),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi": ProbePlan(
                positive=ProbeCase(
                    "frequency-power SQI returns the expected band-power ratio",
                    lambda func: func(signal, 1000.0, 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                    _assert_scalar(0.0),
                ),
                negative=ProbeCase(
                    "frequency-power SQI rejects a non-numeric sampling rate",
                    lambda func: func(signal, "bad", 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi": ProbePlan(
                positive=ProbeCase(
                    "refined-ingest ZZ2018 composite SQI assembles the expected score bundle",
                    lambda func: func(signal, detector_1, detector_2, 1000.0, 50, 64, "simple", 100.0, 0.0, 1.5),
                    _assert_dict_keys({"b_sqi", "f_sqi", "k_sqi"}),
                ),
                negative=ProbeCase(
                    "refined-ingest ZZ2018 composite SQI rejects a non-numeric sampling rate",
                    lambda func: func(signal, detector_1, detector_2, "bad", 50, 64, "simple", 100.0, 0.0, 1.5),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_zz2018_d12.computebeatagreementsqi": ProbePlan(
                positive=ProbeCase(
                    "refined-ingest beat-agreement SQI returns the expected agreement score",
                    lambda func: func(detector_1, detector_2, 1000.0, "simple", 50),
                    _assert_scalar(100.0),
                ),
                negative=ProbeCase(
                    "refined-ingest beat-agreement SQI rejects a non-numeric sampling rate",
                    lambda func: func(detector_1, detector_2, "bad", "simple", 50),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.ecg_zz2018_d12.computefrequencysqi": ProbePlan(
                positive=ProbeCase(
                    "refined-ingest frequency-power SQI returns the expected band-power ratio",
                    lambda func: func(signal, 1000.0, 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                    _assert_scalar(0.0),
                ),
                negative=ProbeCase(
                    "refined-ingest frequency-power SQI rejects a non-numeric sampling rate",
                    lambda func: func(signal, "bad", 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }


    def _biosppy_online_filter_plans() -> dict[str, ProbePlan]:
        coeff_b = np.array([0.5, 0.5], dtype=float)
        coeff_a = np.array([1.0], dtype=float)
        signal = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

        def _invoke_filterstep(func: Callable[..., Any]) -> Any:
            module = importlib.import_module(func.__module__)
            _, state = module.filterstateinit(coeff_b, coeff_a)
            return func(signal, state)

        def _invoke_invalid_filterstep(func: Callable[..., Any]) -> Any:
            module = importlib.import_module(func.__module__)
            _, state = module.filterstateinit(coeff_b, coeff_a)
            return func(None, state)

        return {
            "ageoa.biosppy.online_filter.filterstateinit": ProbePlan(
                positive=ProbeCase(
                    "initialize a chunked BioSPPy OnlineFilter state bundle",
                    lambda func: func(coeff_b, coeff_a),
                    _assert_online_filter_init_state(),
                ),
                negative=ProbeCase(
                    "reject a zero leading denominator coefficient",
                    lambda func: func(coeff_b, np.array([0.0], dtype=float)),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.online_filter.filterstep": ProbePlan(
                positive=ProbeCase(
                    "filter one chunk with a serialized OnlineFilter state",
                    _invoke_filterstep,
                    _assert_online_filter_step_result(),
                ),
                negative=ProbeCase(
                    "reject a missing signal chunk",
                    _invoke_invalid_filterstep,
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.online_filter_codex.filterstateinit": ProbePlan(
                positive=ProbeCase(
                    "initialize a chunked BioSPPy OnlineFilter state bundle",
                    lambda func: func(coeff_b, coeff_a),
                    _assert_online_filter_init_state(),
                ),
                negative=ProbeCase(
                    "reject a zero leading denominator coefficient",
                    lambda func: func(coeff_b, np.array([0.0], dtype=float)),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.online_filter_codex.filterstep": ProbePlan(
                positive=ProbeCase(
                    "filter one chunk with a serialized OnlineFilter state",
                    _invoke_filterstep,
                    _assert_online_filter_step_result(),
                ),
                negative=ProbeCase(
                    "reject a missing signal chunk",
                    _invoke_invalid_filterstep,
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.online_filter_v2.filterstateinit": ProbePlan(
                positive=ProbeCase(
                    "initialize a chunked BioSPPy OnlineFilter state bundle",
                    lambda func: func(coeff_b, coeff_a),
                    _assert_online_filter_init_state(),
                ),
                negative=ProbeCase(
                    "reject a zero leading denominator coefficient",
                    lambda func: func(coeff_b, np.array([0.0], dtype=float)),
                    expect_exception=True,
                ),
                parity_used=True,
            ),
            "ageoa.biosppy.online_filter_v2.filterstep": ProbePlan(
                positive=ProbeCase(
                    "filter one chunk with a serialized OnlineFilter state",
                    _invoke_filterstep,
                    _assert_online_filter_step_result(),
                ),
                negative=ProbeCase(
                    "reject a missing signal chunk",
                    _invoke_invalid_filterstep,
                    expect_exception=True,
                ),
                parity_used=True,
            ),
        }


    plans: dict[str, Any] = {}
    plans.update(_biosppy_detector_plans())
    plans.update(_biosppy_sqi_plans())
    plans.update(_biosppy_online_filter_plans())
    return plans
