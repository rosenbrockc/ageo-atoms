"""Tests for institutional_quant_engine."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.institutional_quant_engine.atoms import market_making_avellaneda\nfrom ageoa.institutional_quant_engine.atoms import almgren_chriss_execution\nfrom ageoa.institutional_quant_engine.atoms import pin_informed_trading\nfrom ageoa.institutional_quant_engine.atoms import limit_order_queue_estimator\n\ndef test_market_making_avellaneda_positive():
    with pytest.raises(NotImplementedError):
        market_making_avellaneda(np.array([1.0]))

def test_market_making_avellaneda_precondition():
    with pytest.raises(icontract.ViolationError):
        market_making_avellaneda(None)

def test_almgren_chriss_execution_positive():
    with pytest.raises(NotImplementedError):
        almgren_chriss_execution(np.array([1.0]))

def test_almgren_chriss_execution_precondition():
    with pytest.raises(icontract.ViolationError):
        almgren_chriss_execution(None)

def test_pin_informed_trading_positive():
    with pytest.raises(NotImplementedError):
        pin_informed_trading(np.array([1.0]))

def test_pin_informed_trading_precondition():
    with pytest.raises(icontract.ViolationError):
        pin_informed_trading(None)

def test_limit_order_queue_estimator_positive():
    with pytest.raises(NotImplementedError):
        limit_order_queue_estimator(np.array([1.0]))

def test_limit_order_queue_estimator_precondition():
    with pytest.raises(icontract.ViolationError):
        limit_order_queue_estimator(None)

