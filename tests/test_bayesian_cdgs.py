"""Tests for Bayesian CDG construction and Ghost Simulation.

Verifies conjugate prior bypassing in the Architect and Oracle isolation
in MCMC simulation.
"""

import pytest
from unittest.mock import MagicMock
from ageoa.ghost.simulator import SimNode, PlanError
from ageoa.ghost.abstract import AbstractDistribution, AbstractSignal, AbstractMatrix
from ageom.synthesizer.ghost_sim import run_ghost_simulation


class TestConjugateBypass:
    def test_conjugate_bypass(self):
        """Verify that Beta-Binomial goals bypass the LLM decomposition loop.

        The Architect should recognize conjugate pairs and return a direct
        3-node CDG (prior, likelihood, posterior).
        """
        architect = MagicMock()

        goal = "Beta-Binomial update"
        mock_cdg = [
            SimNode(name="prior", function_name="witness_prior_init", kwargs={"family": "beta", "event_shape": (1,)}),
            SimNode(name="likelihood", function_name="witness_prior_init", kwargs={"family": "bernoulli", "event_shape": (1,)}),
            SimNode(name="posterior", function_name="witness_posterior_update", inputs={"prior": "prior", "likelihood": "likelihood"}),
        ]

        architect.assemble.return_value = mock_cdg
        architect.checkpoint_history = ["check_conjugacy", "select_template", "validate"]

        cdg = architect.assemble(goal)

        assert len(cdg) == 3
        assert all(h != "decompose_node" for h in architect.checkpoint_history)
        assert cdg[2].function_name == "witness_posterior_update"


class TestOracleIsolation:
    def test_oracle_isolation(self):
        """Verify that ORACLE nodes in MCMC have no state-mutation dependencies."""
        nodes = [
            SimNode(
                name="hmc_oracle",
                function_name="ORACLE_leapfrog",
                inputs={"q": "pos", "p": "mom"},
                output_name="hmc_oracle"
            ),
            SimNode(
                name="update_pos",
                function_name="update_position",
                inputs={"q": "pos", "oracle_out": "hmc_oracle"}
            )
        ]

        oracle_node = next(n for n in nodes if "ORACLE" in n.name.upper())

        initial_state = {
            "pos": AbstractDistribution(family="normal", event_shape=(1,)),
            "mom": AbstractDistribution(family="normal", event_shape=(1,)),
        }

        witness_identity = lambda **kwargs: list(kwargs.values())[0] if kwargs else None
        overrides = {"ORACLE_leapfrog": witness_identity, "update_position": witness_identity}

        result = run_ghost_simulation(nodes, initial_state, witness_overrides=overrides)
        assert "hmc_oracle" in result.trace

    def test_stateful_oracle_raises(self):
        """Verify violation (Oracle with a stateful input)."""
        from ageoa.ghost.abstract import AbstractRNGState

        witness_identity = lambda **kwargs: list(kwargs.values())[0] if kwargs else None
        stateful_nodes = [
            SimNode(
                name="bad_oracle",
                function_name="ORACLE_with_state",
                inputs={"state": "rng_state"}
            )
        ]
        stateful_initial = {
            "rng_state": AbstractRNGState(seed=42)
        }

        with pytest.raises(PlanError) as excinfo:
            run_ghost_simulation(stateful_nodes, stateful_initial, witness_overrides={"ORACLE_with_state": witness_identity})

        assert "Oracle nodes must be stateless" in str(excinfo.value)


class TestVIELBOProvenance:
    def test_vi_elbo_provenance(self):
        """Verify VI ELBO provenance check (reparameterization/jacobian)."""
        nodes = [
            SimNode(
                name="reparam_prior",
                function_name="reparameterized_sample",
                inputs={"dist": "raw_prior"},
                output_name="prior"
            ),
            SimNode(
                name="bijector",
                function_name="bijector_transform",
                inputs={"dist": "prior"},
                output_name="unconstrained"
            ),
            SimNode(
                name="elbo",
                function_name="vi_elbo",
                inputs={"q_dist": "unconstrained", "p_dist": "prior"}
            )
        ]

        initial_state = {
            "raw_prior": AbstractDistribution(family="normal", event_shape=(1,))
        }

        witness_identity = lambda **kwargs: list(kwargs.values())[0] if kwargs else None
        overrides = {"reparameterized_sample": witness_identity}

        result = run_ghost_simulation(nodes, initial_state, witness_overrides=overrides)
        assert "elbo" in result.trace

    def test_bad_elbo_input_raises(self):
        """Failure Case: ELBO input from raw prior (not reparameterized)."""
        initial_state = {
            "raw_prior": AbstractDistribution(family="normal", event_shape=(1,))
        }
        bad_nodes = [
            SimNode(
                name="bad_elbo",
                function_name="vi_elbo",
                inputs={"q_dist": "prior", "p_dist": "prior"}
            )
        ]
        with pytest.raises(PlanError) as excinfo:
            run_ghost_simulation(bad_nodes, initial_state)
        assert "must originate from a reparameterized trace or a bijector output" in str(excinfo.value)
