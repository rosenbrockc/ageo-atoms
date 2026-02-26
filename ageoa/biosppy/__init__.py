"""BioSPPy ECG atoms ingested via the Smart Ingester."""

from .abp_zong.atoms import audio_onset_detection
from .ecg_asi.atoms import thresholdbasedsignalsegmentation
from .ecg_christov.atoms import christovqrsdetect
from .ecg_engzee.atoms import engzee_signal_segmentation
from .ecg_gamboa.atoms import gamboa_segmentation
from .ecg_hamilton.atoms import hamilton_segmentation
from .ecg_zz2018.atoms import (
    calculatecompositesqi_zz2018,
    calculatebeatagreementsqi,
    calculatefrequencypowersqi,
    calculatekurtosissqi,
)
from .emg_abbink.atoms import detect_onsets_with_rest_aware_thresholds
from .emg_bonato.atoms import bonato_onset_detection
from .emg_solnik.atoms import threshold_based_onset_detection
from .pcg_homomorphic.atoms import homomorphic_signal_filtering
from .ppg_elgendi.atoms import detect_signal_onsets_elgendi2013
from .ppg_kavsaoglu.atoms import detectonsetevents

__all__ = [
    "audio_onset_detection",
    "thresholdbasedsignalsegmentation",
    "christovqrsdetect",
    "engzee_signal_segmentation",
    "gamboa_segmentation",
    "hamilton_segmentation",
    "calculatecompositesqi_zz2018",
    "calculatebeatagreementsqi",
    "calculatefrequencypowersqi",
    "calculatekurtosissqi",
    "detect_onsets_with_rest_aware_thresholds",
    "bonato_onset_detection",
    "threshold_based_onset_detection",
    "homomorphic_signal_filtering",
    "detect_signal_onsets_elgendi2013",
    "detectonsetevents",
]
