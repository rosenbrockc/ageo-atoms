"""Tests for pulsar_folding."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.pulsar_folding.atoms import dm_can_brute_force\nfrom ageoa.pulsar_folding.atoms import spline_bandpass_correction\n\ndef test_dm_can_brute_force_positive():
    with pytest.raises(NotImplementedError):
        dm_can_brute_force(np.array([1.0]))

def test_dm_can_brute_force_precondition():
    with pytest.raises(icontract.ViolationError):
        dm_can_brute_force(None)

def test_spline_bandpass_correction_positive():
    with pytest.raises(NotImplementedError):
        spline_bandpass_correction(np.array([1.0]))

def test_spline_bandpass_correction_precondition():
    with pytest.raises(icontract.ViolationError):
        spline_bandpass_correction(None)

