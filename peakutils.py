"""Small compatibility shim for the subset of peakutils used by biosppy."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def indexes(
    y: np.ndarray,
    thres: float = 0.3,
    min_dist: int = 1,
    thres_abs: bool = False,
) -> np.ndarray:
    """Return peak indices compatible with ``peakutils.indexes`` semantics.

    biosppy only relies on thresholded local-maxima detection with a minimum
    distance constraint, so this shim implements that subset via SciPy.
    """

    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0:
        return np.array([], dtype=int)

    distance = max(int(min_dist), 1)
    if thres_abs:
        threshold = float(thres)
    else:
        ymin = float(np.min(values))
        ymax = float(np.max(values))
        threshold = ymin + float(thres) * (ymax - ymin)

    candidate_peaks, properties = find_peaks(values, height=threshold)
    if candidate_peaks.size == 0:
        return np.array([], dtype=int)

    heights = properties.get("peak_heights", values[candidate_peaks])
    order = np.argsort(heights)[::-1]
    keep: list[int] = []
    for peak in candidate_peaks[order]:
        if all(abs(int(peak) - existing) > distance for existing in keep):
            keep.append(int(peak))

    keep.sort()
    return np.asarray(keep, dtype=int)
