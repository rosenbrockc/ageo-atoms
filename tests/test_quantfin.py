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

    def test_net_yc_dump_preserves_subclass_fields(self):
        c1 = qf.FlatCurve(rate=0.05)
        c2 = qf.FlatCurve(rate=0.02)
        net = qf.NetYC(yc1=c1, yc2=c2)
        dump = net.model_dump()
        assert np.isclose(dump["yc1"]["rate"], 0.05)
        assert np.isclose(dump["yc2"]["rate"], 0.02)

    def test_contingent_claim_dump_json_excludes_callables(self):
        def payout(_obs):
            return qf.CashFlow(time=1.0, amount=100.0)

        proc = qf.CCProcessor(
            monitor_time=0.0,
            payout_func_names=["terminal_payout"],
            payout_funcs=[payout],
        )
        claim = qf.ContingentClaim(processors=[proc])
        dumped = claim.model_dump()
        assert dumped["processors"][0]["payout_func_names"] == ["terminal_payout"]
        assert "payout_funcs" not in dumped["processors"][0]
        assert "terminal_payout" in claim.model_dump_json()

class TestQuantFinMonteCarlo:
    def test_run_simulation_anti_positive(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        seed = 42
        trials = 100

        # A mock simulator that returns 1.0 for antithetic=True and 2.0 for antithetic=False
        def mock_sim(m, c, rng, t, anti):
            del m, c, rng, t
            return 1.0 if anti else 2.0

        qf.register_simulator("const_avg", mock_sim)
        result = qf.run_simulation_anti(model, claim, seed, trials, "const_avg")

        # The result should be the average: (1.0 + 2.0) / 2 = 1.5
        assert np.isclose(result, 1.5)

    def test_quick_sim_anti_positive(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        trials = 50

        draws: list[int] = []

        def mock_sim(m, c, rng, t, anti):
            del m, c, t
            draws.append(int(rng.integers(0, 10_000)))
            return 10.0 if anti else 20.0

        qf.register_simulator("seed_probe", mock_sim)
        result = qf.quick_sim_anti(model, claim, trials, "seed_probe")
        assert np.isclose(result, 15.0)
        assert len(draws) == 2
        assert draws[0] == draws[1]

    def test_seeded_simulator_is_reproducible(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()

        def random_sim(m, c, rng, t, anti):
            del m, c, t, anti
            return float(rng.standard_normal())

        qf.register_simulator("normal_draw", random_sim)
        first = qf.run_simulation(model, claim, seed=123, trials=10, anti=False, simulator_name="normal_draw")
        second = qf.run_simulation(model, claim, seed=123, trials=10, anti=False, simulator_name="normal_draw")
        assert first == second

    def test_run_simulation_anti_requires_even_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()

        def mock_sim(m, c, rng, t, anti):
            del m, c, rng, t, anti
            return 0.0

        qf.register_simulator("zero_even", mock_sim)
        with pytest.raises(icontract.ViolationError, match="even for antithetic variates"):
            qf.run_simulation_anti(model, claim, 42, 11, "zero_even")

    def test_run_simulation_anti_requires_positive_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()

        def mock_sim(m, c, rng, t, anti):
            del m, c, rng, t, anti
            return 0.0

        qf.register_simulator("zero_positive", mock_sim)
        with pytest.raises(icontract.ViolationError):
            qf.run_simulation_anti(model, claim, 42, -10, "zero_positive")

    def test_quick_sim_anti_requires_even_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()

        def mock_sim(m, c, rng, t, anti):
            del m, c, rng, t, anti
            return 0.0

        qf.register_simulator("zero_quick_even", mock_sim)
        with pytest.raises(icontract.ViolationError, match="even for antithetic variates"):
            qf.quick_sim_anti(model, claim, 11, "zero_quick_even")

    def test_quick_sim_anti_requires_positive_trials(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()

        def mock_sim(m, c, rng, t, anti):
            del m, c, rng, t, anti
            return 0.0

        qf.register_simulator("zero_quick_positive", mock_sim)
        with pytest.raises(icontract.ViolationError):
            qf.quick_sim_anti(model, claim, -10, "zero_quick_positive")

    def test_requires_registered_simulator(self):
        model = qf.DiscretizeModel()
        claim = qf.ContingentClaim()
        with pytest.raises(icontract.ViolationError, match="registered simulator"):
            qf.run_simulation_anti(model, claim, 42, 10, "missing_sim")
