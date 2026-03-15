"""Auto-generated parity tests for biosppy atoms.

DO NOT EDIT — regenerate with: python scripts/generate_parity_tests.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# Import helpers from test_parity without requiring tests to be a package
_spec = importlib.util.spec_from_file_location(
    "test_parity", Path(__file__).parent / "test_parity.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_assert_outputs_match = _mod._assert_outputs_match
_deserialize = _mod._deserialize
_import_atom = _mod._import_atom

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "biosppy"

class TestThresholdbasedsignalsegmentation:
    FIXTURE = FIXTURES_DIR / "ecg_asi/thresholdbasedsignalsegmentation.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_asi:thresholdbasedsignalsegmentation")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestAsiSignalSegmenter:
    FIXTURE = FIXTURES_DIR / "ecg_asi_d12/asi_signal_segmenter.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_asi_d12:asi_signal_segmenter")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestChristovqrsdetect:
    FIXTURE = FIXTURES_DIR / "ecg_christov/christovqrsdetect.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_christov:christovqrsdetect")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestChristovQrsSegmenter:
    FIXTURE = FIXTURES_DIR / "ecg_christov_d12/christov_qrs_segmenter.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_christov_d12:christov_qrs_segmenter")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestEngzeeSignalSegmentation:
    FIXTURE = FIXTURES_DIR / "ecg_engzee/engzee_signal_segmentation.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_engzee:engzee_signal_segmentation")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestEngzeeQrsSegmentation:
    FIXTURE = FIXTURES_DIR / "ecg_engzee_d12/engzee_qrs_segmentation.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_engzee_d12:engzee_qrs_segmentation")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestGamboaSegmentation:
    FIXTURE = FIXTURES_DIR / "ecg_gamboa/gamboa_segmentation.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_gamboa:gamboa_segmentation")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestGamboaSegmenter:
    FIXTURE = FIXTURES_DIR / "ecg_gamboa_d12/gamboa_segmenter.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_gamboa_d12:gamboa_segmenter")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestCalculatekurtosissqi:
    FIXTURE = FIXTURES_DIR / "ecg_zz2018/calculatekurtosissqi.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_zz2018:calculatekurtosissqi")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestComputekurtosissqi:
    FIXTURE = FIXTURES_DIR / "ecg_zz2018_d12/computekurtosissqi.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ecg_zz2018_d12:computekurtosissqi")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestHomomorphicSegmentation:
    FIXTURE = FIXTURES_DIR / "pcg_homomorphic/homomorphic_segmentation.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/pcg_homomorphic:homomorphic_segmentation")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestHomomorphicSignalFiltering:
    FIXTURE = FIXTURES_DIR / "pcg_homomorphic/homomorphic_signal_filtering.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/pcg_homomorphic:homomorphic_signal_filtering")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestHomomorphicfilter:
    FIXTURE = FIXTURES_DIR / "pcg_homomorphic_d12/homomorphicfilter.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(1))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/pcg_homomorphic_d12:homomorphicfilter")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

class TestDetectonsetevents:
    FIXTURE = FIXTURES_DIR / "ppg_kavsaoglu/detectonsetevents.json"

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(self.FIXTURE) as f:
            self.cases = json.load(f)

    @pytest.mark.parametrize("idx", range(2))
    def test_parity(self, idx):
        case = self.cases[idx]
        atom_fn = _import_atom("biosppy/ppg_kavsaoglu:detectonsetevents")
        inputs = _deserialize(case["inputs"])
        expected = _deserialize(case["output"])
        try:
            result = atom_fn(**inputs)
        except NotImplementedError:
            pytest.skip("stub")
        _assert_outputs_match(result, expected)

