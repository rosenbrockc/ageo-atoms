"""Tests for pronto."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.pronto.atoms import rbis_state_estimation\n\ndef test_rbis_state_estimation_positive():
    with pytest.raises(NotImplementedError):
        rbis_state_estimation(np.array([1.0]))

def test_rbis_state_estimation_precondition():
    with pytest.raises(icontract.ViolationError):
        rbis_state_estimation(None)

