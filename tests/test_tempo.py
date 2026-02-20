"""Tests for ageoa.tempo atoms."""

import numpy as np
import pytest
import icontract

import ageoa.tempo as ag_tempo


class TestTempoTdb:
    """Tests for the offset_tt2tdb and offset_tai2tdb atoms."""

    def test_positive_scalar(self):
        seconds = 0.0
        tt_tdb = ag_tempo.offset_tt2tdb(seconds)
        assert isinstance(tt_tdb, float)
        # Verify that the offset is computed properly
        assert np.isfinite(tt_tdb)
        
        tai_tdb = ag_tempo.offset_tai2tdb(seconds)
        assert isinstance(tai_tdb, float)
        assert np.isfinite(tai_tdb)
        
    def test_positive_array(self):
        seconds = np.array([0.0, 100.0, 200.0])
        tt_tdb = ag_tempo.offset_tt2tdb(seconds)
        assert isinstance(tt_tdb, np.ndarray)
        assert tt_tdb.shape == seconds.shape
        assert tt_tdb.dtype == np.float64
        
        tai_tdb = ag_tempo.offset_tai2tdb(seconds)
        assert isinstance(tai_tdb, np.ndarray)
        assert tai_tdb.shape == seconds.shape
        assert tai_tdb.dtype == np.float64

    def test_require_numeric_types(self):
        with pytest.raises(NotImplementedError):
            ag_tempo.offset_tt2tdb("100")
            
        with pytest.raises(NotImplementedError):
            ag_tempo.offset_tai2tdb("100")

    def test_precision_consistency(self):
        # Using python floats vs numpy f64
        sec1 = 50.0
        sec2 = np.float64(50.0)
        
        tt_tdb1 = ag_tempo.offset_tt2tdb(sec1)
        tt_tdb2 = ag_tempo.offset_tt2tdb(sec2)
        assert tt_tdb1 == tt_tdb2
