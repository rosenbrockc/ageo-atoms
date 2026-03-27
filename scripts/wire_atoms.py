#!/usr/bin/env python3
"""Wire stub atoms to their upstream implementations.

Reads atom files, replaces `raise NotImplementedError(...)` with delegation
calls to the upstream function specified in WIRING_TABLE below.

Usage::

    python scripts/wire_atoms.py          # wire all entries
    python scripts/wire_atoms.py --dry    # preview changes without writing
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGEOA = ROOT / "ageoa"

# ---------------------------------------------------------------------------
# Wiring table: (atom_file_path, function_name, upstream_import, upstream_call)
#
# upstream_import is added after existing imports.
# upstream_call replaces `raise NotImplementedError(...)`.
# ---------------------------------------------------------------------------

WIRING_TABLE: list[dict] = [
    # ── BioSPPy ECG ──────────────────────────────────────────────────────
    {
        "file": "biosppy/ecg_hamilton/atoms.py",
        "func": "hamilton_segmentation",
        "import": "from biosppy.signals.ecg import hamilton_segmenter",
        "call": "return hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)",
    },
    {
        "file": "biosppy/ecg_christov/atoms.py",
        "func": "christovqrsdetect",
        "import": "from biosppy.signals.ecg import christov_segmenter",
        "call": "return christov_segmenter(signal=signal, sampling_rate=sampling_rate)",
    },
    {
        "file": "biosppy/ecg_engzee/atoms.py",
        "func": "engzee_signal_segmentation",
        "import": "from biosppy.signals.ecg import engzee_segmenter",
        "call": "return engzee_segmenter(signal=signal, sampling_rate=sampling_rate, threshold=threshold)",
    },
    {
        "file": "biosppy/ecg_gamboa/atoms.py",
        "func": "gamboa_segmentation",
        "import": "from biosppy.signals.ecg import gamboa_segmenter",
        "call": "return gamboa_segmenter(signal=signal, sampling_rate=sampling_rate, tol=tol)",
    },
    {
        "file": "biosppy/ecg_detectors.py",
        "func": "thresholdbasedsignalsegmentation",
        "import": "from biosppy.signals.ecg import ASI_segmenter",
        "call": "return ASI_segmenter(signal=signal, sampling_rate=sampling_rate, Pth=Pth)",
    },
    # ── BioSPPy ECG SQI (zz2018) ─────────────────────────────────────────
    {
        "file": "biosppy/ecg_zz2018/atoms.py",
        "func": "calculatecompositesqi_zz2018",
        "import": "from biosppy.signals.ecg import ZZ2018",
        "call": "return ZZ2018(signal=signal, detector_1=detector_1, detector_2=detector_2, fs=fs, search_window=search_window, nseg=nseg, mode=mode)",
    },
    {
        "file": "biosppy/ecg_zz2018/atoms.py",
        "func": "calculatebeatagreementsqi",
        "import": "from biosppy.signals.ecg import bSQI",
        "call": "return bSQI(detector_1=detector_1, detector_2=detector_2, fs=fs, mode=mode, search_window=search_window)",
    },
    {
        "file": "biosppy/ecg_zz2018/atoms.py",
        "func": "calculatefrequencypowersqi",
        "import": "from biosppy.signals.ecg import fSQI",
        "call": "return fSQI(ecg_signal=ecg_signal, fs=fs, nseg=nseg, num_spectrum=num_spectrum, dem_spectrum=dem_spectrum, mode=mode)",
    },
    {
        "file": "biosppy/ecg_zz2018/atoms.py",
        "func": "calculatekurtosissqi",
        "import": "from biosppy.signals.ecg import kSQI",
        "call": "return kSQI(signal=signal, fisher=fisher)",
    },
    # ── BioSPPy EMG ──────────────────────────────────────────────────────
    {
        "file": "biosppy/emg_abbink/atoms.py",
        "func": "detect_onsets_with_rest_aware_thresholds",
        "import": "from biosppy.signals.emg import abbink_onset_detector",
        "call": "return abbink_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, size=size, alarm_size=alarm_size, threshold=threshold, transition_threshold=transition_threshold)",
    },
    {
        "file": "biosppy/emg_bonato/atoms.py",
        "func": "bonato_onset_detection",
        "import": "from biosppy.signals.emg import bonato_onset_detector",
        "call": "return bonato_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, threshold=threshold, active_state_duration=active_state_duration, samples_above_fail=samples_above_fail, fail_size=fail_size)",
    },
    {
        "file": "biosppy/emg_solnik/atoms.py",
        "func": "threshold_based_onset_detection",
        "import": "from biosppy.signals.emg import solnik_onset_detector",
        "call": "return solnik_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, threshold=threshold, active_state_duration=active_state_duration)",
    },
    # ── BioSPPy PCG ──────────────────────────────────────────────────────
    {
        "file": "biosppy/pcg_homomorphic/atoms.py",
        "func": "homomorphic_signal_filtering",
        "import": "from biosppy.signals.pcg import homomorphic_filter",
        "call": "return homomorphic_filter(signal=signal, sampling_rate=sampling_rate)",
    },
    # ── BioSPPy PPG ──────────────────────────────────────────────────────
    {
        "file": "biosppy/ppg_elgendi/atoms.py",
        "func": "detect_signal_onsets_elgendi2013",
        "import": "from biosppy.signals.ppg import find_onsets_elgendi2013",
        "call": "return find_onsets_elgendi2013(signal=signal, sampling_rate=sampling_rate, peakwindow=peakwindow, beatwindow=beatwindow, beatoffset=beatoffset, mindelay=mindelay)",
    },
    {
        "file": "biosppy/ppg_kavsaoglu/atoms.py",
        "func": "detectonsetevents",
        "import": "from biosppy.signals.ppg import find_onsets_kavsaoglu2016",
        "call": "return find_onsets_kavsaoglu2016(signal=signal, sampling_rate=sampling_rate, alpha=alpha, k=k, init_bpm=init_bpm, min_delay=min_delay, max_BPM=max_BPM)",
    },
    # ── BioSPPy ABP ──────────────────────────────────────────────────────
    {
        "file": "biosppy/abp_zong/atoms.py",
        "func": "audio_onset_detection",
        "import": "from biosppy.signals.abp import find_onsets_zong2003",
        "call": "return find_onsets_zong2003(signal=signal, sampling_rate=sampling_rate, sm_size=sm_size, size=size, alpha=alpha, wrange=wrange, d1_th=d1_th, d2_th=d2_th)",
    },
    # ── NeuroKit2 ─────────────────────────────────────────────────────────
    {
        "file": "neurokit2/atoms.py",
        "func": "averageqrstemplate",
        "import": "from neurokit2.ecg.ecg_quality import _ecg_quality_averageQRS",
        "call": "return _ecg_quality_averageQRS(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)",
    },
    {
        "file": "neurokit2/atoms.py",
        "func": "zhao2018hrvanalysis",
        "import": "from neurokit2.ecg.ecg_quality import _ecg_quality_zhao2018",
        "call": "return _ecg_quality_zhao2018(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, window=window, mode=mode)",
    },
    # ── Skyfield ──────────────────────────────────────────────────────────
    {
        "file": "skyfield/atoms.py",
        "func": "calculate_vector_angle",
        "import": "from skyfield.functions import angle_between",
        "call": "return angle_between(u, v)",
    },
    {
        "file": "skyfield/atoms.py",
        "func": "compute_spherical_coordinate_rates",
        "import": "from skyfield.functions import _to_spherical_and_rates",
        "call": "return _to_spherical_and_rates(r, v)",
    },
    # ── E2E-PPG ───────────────────────────────────────────────────────────
    {
        "file": "e2e_ppg/kazemi_wrapper/atoms.py",
        "func": "signalarraynormalization",
        "import": "from kazemi_peak_detection import normalize",
        "call": "return normalize(arr)",
    },
    {
        "file": "e2e_ppg/heart_cycle.py",
        "func": "detect_heart_cycles",
        "import": "from ppg_sqa import heart_cycle_detection",
        "call": "return heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate)",
    },
    {
        "file": "e2e_ppg/template_matching/atoms.py",
        "func": "templatefeaturecomputation",
        "import": "from ppg_sqa import template_matching_features",
        "call": "return template_matching_features(hc=hc)",
    },
    {
        "file": "e2e_ppg/reconstruction/atoms.py",
        "func": "windowed_signal_reconstruction",
        "import": "from ppg_reconstruction import reconstruction",
        "call": "return reconstruction(sig=sig, clean_indices=clean_indices, noisy_indices=noisy_indices, sampling_rate=sampling_rate, filter_signal=filter_signal)",
    },
    {
        "file": "e2e_ppg/reconstruction/atoms.py",
        "func": "gan_patch_reconstruction",
        "import": "from ppg_reconstruction import gan_rec",
        "call": "return gan_rec(ppg_clean=ppg_clean, noise=noise, sampling_rate=sampling_rate, generator=generator, device=device)",
    },
    {
        "file": "e2e_ppg/gan_reconstruction.py",
        "func": "generatereconstructedppg",
        "import": "from ppg_reconstruction import gan_rec",
        "call": "return gan_rec(ppg_clean=ppg_clean, noise=noise, sampling_rate=sampling_rate, generator=generator, device=device)",
    },
    {
        "file": "e2e_ppg/gan_reconstruction.py",
        "func": "gan_reconstruction",
        "import": "from ppg_reconstruction import gan_rec",
        "call": "return gan_rec(ppg_clean=ppg_clean, noise=noise, sampling_rate=sampling_rate, generator=generator, device=device)",
    },
    {
        "file": "e2e_ppg/heart_cycle.py",
        "func": "heart_cycle_detection",
        "import": "from ppg_sqa import heart_cycle_detection as _heart_cycle_detection",
        "call": "return _heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate)",
    },
    {
        "file": "e2e_ppg/kazemi_wrapper_d12/atoms.py",
        "func": "normalizesignal",
        "import": "from kazemi_peak_detection import normalize",
        "call": "return normalize(arr)",
    },
]


# ---------------------------------------------------------------------------
# D12 variants — same upstream as non-d12, scan and auto-generate
# ---------------------------------------------------------------------------

# Map d12 atoms to their non-d12 upstream (same BioSPPy function)
D12_WIRING: list[dict] = [
    {
        "file": "biosppy/ecg_hamilton_d12/atoms.py",
        "func": "hamilton_segmenter",
        "import": "from biosppy.signals.ecg import hamilton_segmenter as _hamilton_segmenter",
        "call": "return _hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)",
    },
    {
        "file": "biosppy/ecg_christov_d12/atoms.py",
        "func": "christov_qrs_segmenter",
        "import": "from biosppy.signals.ecg import christov_segmenter",
        "call": "return christov_segmenter(signal=signal, sampling_rate=sampling_rate)",
    },
    {
        "file": "biosppy/ecg_engzee_d12/atoms.py",
        "func": "engzee_qrs_segmentation",
        "import": "from biosppy.signals.ecg import engzee_segmenter",
        "call": "return engzee_segmenter(signal=signal, sampling_rate=sampling_rate, threshold=threshold)",
    },
    {
        "file": "biosppy/ecg_gamboa_d12/atoms.py",
        "func": "gamboa_segmenter",
        "import": "from biosppy.signals.ecg import gamboa_segmenter as _gamboa_segmenter",
        "call": "return _gamboa_segmenter(signal=signal, sampling_rate=sampling_rate, tol=tol)",
    },
    {
        "file": "biosppy/ecg_detectors.py",
        "func": "asi_signal_segmenter",
        "import": "from biosppy.signals.ecg import ASI_segmenter",
        "call": "return ASI_segmenter(signal=signal, sampling_rate=sampling_rate, Pth=Pth)",
    },
    {
        "file": "biosppy/ecg_zz2018_d12/atoms.py",
        "func": "assemblezz2018sqi",
        "import": "",
        "call": 'return {"b_sqi": b_sqi, "f_sqi": f_sqi, "k_sqi": k_sqi}',
    },
    {
        "file": "biosppy/ecg_zz2018_d12/atoms.py",
        "func": "computebeatagreementsqi",
        "import": "from biosppy.signals.ecg import bSQI",
        "call": "return bSQI(detector_1=detector_1, detector_2=detector_2, fs=fs, mode=mode, search_window=search_window)",
    },
    {
        "file": "biosppy/ecg_zz2018_d12/atoms.py",
        "func": "computefrequencysqi",
        "import": "from biosppy.signals.ecg import fSQI",
        "call": "return fSQI(ecg_signal=ecg_signal, fs=fs, nseg=nseg, num_spectrum=num_spectrum, dem_spectrum=dem_spectrum, mode=mode)",
    },
    {
        "file": "biosppy/ecg_zz2018_d12/atoms.py",
        "func": "computekurtosissqi",
        "import": "from biosppy.signals.ecg import kSQI",
        "call": "return kSQI(signal=signal, fisher=fisher)",
    },
    {
        "file": "biosppy/emg_solnik_d12/atoms.py",
        "func": "solnik_onset_detect",
        "import": "from biosppy.signals.emg import solnik_onset_detector",
        "call": "return solnik_onset_detector(signal=signal, rest=rest, sampling_rate=sampling_rate, threshold=threshold, active_state_duration=active_state_duration)",
    },
    {
        "file": "biosppy/pcg_homomorphic_d12/atoms.py",
        "func": "homomorphicfilter",
        "import": "from biosppy.signals.pcg import homomorphic_filter",
        "call": "return homomorphic_filter(signal=signal, sampling_rate=sampling_rate)",
    },
]

ALL_WIRING = WIRING_TABLE + D12_WIRING


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _find_func_body(content: str, func_name: str) -> tuple[int, int, str] | None:
    """Find the line range of a function's NotImplementedError raise.

    Returns (start_offset, end_offset, matched_text) or None.
    """
    # Match: raise NotImplementedError(...) inside the named function
    # We look for the raise statement that follows the def
    pattern = rf'(def {re.escape(func_name)}\(.*?\).*?:\n(?:.*\n)*?)([ \t]+raise NotImplementedError\([^\)]*\))'
    m = re.search(pattern, content)
    if m:
        raise_text = m.group(2)
        start = m.start(2)
        end = m.end(2)
        return start, end, raise_text

    # Simpler fallback: just find the raise within a reasonable distance of the def
    func_def_match = re.search(rf'^def {re.escape(func_name)}\(', content, re.MULTILINE)
    if not func_def_match:
        return None

    # Search for NotImplementedError after the def
    rest = content[func_def_match.start():]
    raise_match = re.search(r'([ \t]+)raise NotImplementedError\([^\)]*\)', rest)
    if raise_match:
        abs_start = func_def_match.start() + raise_match.start()
        abs_end = func_def_match.start() + raise_match.end()
        return abs_start, abs_end, raise_match.group(0)

    return None


def _add_import(content: str, import_line: str) -> str:
    """Add an import line after existing imports, avoiding duplicates."""
    if not import_line or import_line in content:
        return content

    # Find last import line
    lines = content.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('import ') or stripped.startswith('from ')
                or stripped.startswith('#') or stripped == ''
                or stripped.startswith('"""') or stripped.startswith("'''")):
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import_idx = i

    lines.insert(last_import_idx + 1, import_line)
    return '\n'.join(lines)


def wire_atom(entry: dict, dry_run: bool = False) -> bool:
    """Wire a single atom. Returns True if modified."""
    file_path = AGEOA / entry["file"]
    if not file_path.exists():
        print(f"  SKIP {entry['file']} — file not found")
        return False

    content = file_path.read_text()
    func_name = entry["func"]

    # Check if already wired
    if f"def {func_name}(" in content:
        match = _find_func_body(content, func_name)
        if match is None:
            print(f"  SKIP {entry['file']}:{func_name} — no NotImplementedError found (already wired?)")
            return False
    else:
        print(f"  SKIP {entry['file']}:{func_name} — function not found")
        return False

    start, end, raise_text = match
    indent = re.match(r'[ \t]*', raise_text).group(0)

    # Replace raise with delegation call
    new_content = content[:start] + indent + entry["call"] + content[end:]

    # Add import
    new_content = _add_import(new_content, entry["import"])

    if dry_run:
        print(f"  WIRE {entry['file']}:{func_name}")
        print(f"       - {raise_text.strip()}")
        print(f"       + {entry['call']}")
        return True

    file_path.write_text(new_content)
    print(f"  WIRE {entry['file']}:{func_name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Wire stub atoms to upstream implementations")
    parser.add_argument("--dry", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    wired = 0
    skipped = 0
    for entry in ALL_WIRING:
        if wire_atom(entry, dry_run=args.dry):
            wired += 1
        else:
            skipped += 1

    print(f"\nDone: {wired} wired, {skipped} skipped")


if __name__ == "__main__":
    main()
