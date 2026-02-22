"""Tests for institutional_quant_engine."""

import pytest
import numpy as np
import icontract
from ageoa.institutional_quant_engine.atoms import (
    market_making_avellaneda,
    almgren_chriss_execution,
    pin_informed_trading,
    limit_order_queue_estimator,
)


class TestMarketMakingAvellaneda:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            market_making_avellaneda(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            market_making_avellaneda(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            market_making_avellaneda(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            market_making_avellaneda(np.array([np.nan]))


class TestAlmgrenChrissExecution:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            almgren_chriss_execution(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            almgren_chriss_execution(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            almgren_chriss_execution(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            almgren_chriss_execution(np.array([np.inf]))


class TestPINInformedTrading:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            pin_informed_trading(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            pin_informed_trading(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            pin_informed_trading(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            pin_informed_trading(np.array([np.nan]))


class TestLimitOrderQueueEstimator:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            limit_order_queue_estimator(np.array([1.0, 2.0, 3.0]))

    def test_precondition_none(self):
        with pytest.raises(icontract.ViolationError):
            limit_order_queue_estimator(None)

    def test_precondition_empty(self):
        with pytest.raises(icontract.ViolationError):
            limit_order_queue_estimator(np.array([]))

    def test_precondition_non_finite(self):
        with pytest.raises(icontract.ViolationError):
            limit_order_queue_estimator(np.array([np.inf]))
