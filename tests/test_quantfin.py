"""Tests for ageoa.quantfin atoms."""

import numpy as np
import pytest
import icontract

import ageoa.quantfin as qf


class TestQuantFinModels:
    def test_flat_curve(self):
        curve = qf.FlatCurve(rate=0.05)
        # Check discount factor at t=1 is exp(-0.05 * 1)
        assert np.isclose(curve.disc(1.0), np.exp(-0.05))

    def test_net_yc(self):
        c1 = qf.FlatCurve(rate=0.05)
        c2 = qf.FlatCurve(rate=0.02)
        net = qf.NetYC(yc1=c1, yc2=c2)
        # Check discount factor
        assert np.isclose(net.disc(1.0), np.exp(-0.05) / np.exp(-0.02))

    def test_forward_rate(self):
        curve = qf.FlatCurve(rate=0.05)
        # Forward rate of flat curve is just the rate
        fwd = curve.forward(1.0, 2.0)
        assert np.isclose(fwd, 0.05)

    def test_spot_rate(self):
        curve = qf.FlatCurve(rate=0.05)
        spot = curve.spot(2.0)
        assert np.isclose(spot, 0.05)

class TestQuantFinMonteCarlo:
    def test_run_simulation_anti_positive(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        seed = 42
        trials = 100
        
        # A mock simulator that returns 1.0 for antithetic=True and 2.0 for antithetic=False
        def mock_sim(m, c, s, t, anti):
            return 1.0 if anti else 2.0
            
        result = qf.run_simulation_anti(model, claim, seed, trials, mock_sim)
        
        # The result should be the average: (1.0 + 2.0) / 2 = 1.5
        assert np.isclose(result, 1.5)

    def test_quick_sim_anti_positive(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        trials = 50
        
        def mock_sim(m, c, s, t, anti):
            assert s == 500  # Default seed
            return 10.0 if anti else 20.0
            
        result = qf.quick_sim_anti(model, claim, trials, mock_sim)
        assert np.isclose(result, 15.0)

    def test_run_simulation_anti_requires_even_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        
        def mock_sim(m, c, s, t, anti): return 0.0
            
        with pytest.raises(icontract.ViolationError, match="even for antithetic variates"):
            qf.run_simulation_anti(model, claim, 42, 11, mock_sim)

    def test_run_simulation_anti_requires_positive_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        
        def mock_sim(m, c, s, t, anti): return 0.0
            
        with pytest.raises(icontract.ViolationError):
            qf.run_simulation_anti(model, claim, 42, -10, mock_sim)

    def test_quick_sim_anti_requires_even_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        
        def mock_sim(m, c, s, t, anti): return 0.0
            
        with pytest.raises(icontract.ViolationError, match="even for antithetic variates"):
            qf.quick_sim_anti(model, claim, 11, mock_sim)

    def test_quick_sim_anti_requires_positive_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        
        def mock_sim(m, c, s, t, anti): return 0.0
            
        with pytest.raises(icontract.ViolationError):
            qf.quick_sim_anti(model, claim, -10, mock_sim)
