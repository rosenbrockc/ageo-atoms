"""Tests for e2e_ppg."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.e2e_ppg.atoms import kazemi_peak_detection\nfrom ageoa.e2e_ppg.atoms import ppg_reconstruction\nfrom ageoa.e2e_ppg.atoms import ppg_sqa\n\ndef test_kazemi_peak_detection_positive():
    with pytest.raises(NotImplementedError):
        kazemi_peak_detection(np.array([1.0]))

def test_kazemi_peak_detection_precondition():
    with pytest.raises(icontract.ViolationError):
        kazemi_peak_detection(None)

def test_ppg_reconstruction_positive():
    with pytest.raises(NotImplementedError):
        ppg_reconstruction(np.array([1.0]))

def test_ppg_reconstruction_precondition():
    with pytest.raises(icontract.ViolationError):
        ppg_reconstruction(None)

def test_ppg_sqa_positive():
    with pytest.raises(NotImplementedError):
        ppg_sqa(np.array([1.0]))

def test_ppg_sqa_precondition():
    with pytest.raises(icontract.ViolationError):
        ppg_sqa(None)

