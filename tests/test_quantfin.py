"""Tests for quantfin."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.quantfin.atoms import functional_monte_carlo\nfrom ageoa.quantfin.atoms import volatility_surface_modeling\n\ndef test_functional_monte_carlo_positive():
    with pytest.raises(NotImplementedError):
        functional_monte_carlo(np.array([1.0]))

def test_functional_monte_carlo_precondition():
    with pytest.raises(icontract.ViolationError):
        functional_monte_carlo(None)

def test_volatility_surface_modeling_positive():
    with pytest.raises(NotImplementedError):
        volatility_surface_modeling(np.array([1.0]))

def test_volatility_surface_modeling_precondition():
    with pytest.raises(icontract.ViolationError):
        volatility_surface_modeling(None)

