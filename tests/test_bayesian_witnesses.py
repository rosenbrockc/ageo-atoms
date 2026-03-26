"""Tests for Bayesian abstract types and ghost witnesses."""

from __future__ import annotations

import pytest

from ageoa.ghost.abstract import (
    AbstractArray,
    AbstractDistribution,
    AbstractMCMCTrace,
    AbstractRNGState,
    AbstractScalar,
    CONJUGATE_PAIRS,
    DISTRIBUTION_FAMILIES,
)
from ageoa.ghost.witnesses import (
    witness_log_prob,
    witness_mcmc_step,
    witness_posterior_update,
    witness_prior_init,
    witness_vi_elbo,
)


# ---------------------------------------------------------------------------
# AbstractDistribution
# ---------------------------------------------------------------------------


class TestAbstractDistribution:
    def test_create_normal(self):
        d = AbstractDistribution(family="normal", event_shape=(3,))
        assert d.family == "normal"
        assert d.event_shape == (3,)
        assert d.batch_shape == ()
        assert not d.is_discrete

    def test_create_categorical(self):
        d = AbstractDistribution(
            family="categorical", event_shape=(10,), is_discrete=True
        )
        assert d.is_discrete

    def test_assert_family_pass(self):
        d = AbstractDistribution(family="normal", event_shape=(1,))
        d.assert_family("normal")  # should not raise

    def test_assert_family_fail(self):
        d = AbstractDistribution(family="normal", event_shape=(1,))
        with pytest.raises(ValueError, match="family mismatch"):
            d.assert_family("gamma")

    def test_assert_event_shape_pass(self):
        d = AbstractDistribution(family="normal", event_shape=(3, 2))
        d.assert_event_shape((3, 2))

    def test_assert_event_shape_fail(self):
        d = AbstractDistribution(family="normal", event_shape=(3,))
        with pytest.raises(ValueError, match="Event shape mismatch"):
            d.assert_event_shape((5,))

    def test_assert_conjugate_pass(self):
        prior = AbstractDistribution(family="beta", event_shape=(1,))
        likelihood = AbstractDistribution(family="bernoulli", event_shape=(1,))
        prior.assert_conjugate_to(likelihood)  # should not raise

    def test_assert_conjugate_fail(self):
        prior = AbstractDistribution(family="normal", event_shape=(1,))
        likelihood = AbstractDistribution(family="poisson", event_shape=(1,))
        with pytest.raises(ValueError, match="not a conjugate prior"):
            prior.assert_conjugate_to(likelihood)

    def test_support_bounds(self):
        d = AbstractDistribution(
            family="beta",
            event_shape=(1,),
            support_lower=0.0,
            support_upper=1.0,
        )
        assert d.support_lower == 0.0
        assert d.support_upper == 1.0

    def test_batch_shape(self):
        d = AbstractDistribution(
            family="normal", event_shape=(3,), batch_shape=(100,)
        )
        assert d.batch_shape == (100,)


# ---------------------------------------------------------------------------
# AbstractRNGState
# ---------------------------------------------------------------------------


class TestAbstractRNGState:
    def test_create(self):
        rng = AbstractRNGState(seed=42)
        assert rng.seed == 42
        assert rng.consumed == 0
        assert not rng.is_split

    def test_advance(self):
        rng = AbstractRNGState(seed=42)
        rng2 = rng.advance(5)
        assert rng2.consumed == 5
        assert rng.consumed == 0  # original unchanged

    def test_advance_negative_raises(self):
        rng = AbstractRNGState(seed=42)
        with pytest.raises(ValueError, match="must be positive"):
            rng.advance(-1)

    def test_advance_zero_raises(self):
        rng = AbstractRNGState(seed=42)
        with pytest.raises(ValueError, match="must be positive"):
            rng.advance(0)

    def test_split(self):
        rng = AbstractRNGState(seed=42)
        a, b = rng.split()
        assert a.is_split
        assert b.is_split
        assert a.seed != b.seed  # different keys

    def test_advance_cumulative(self):
        rng = AbstractRNGState(seed=0)
        rng = rng.advance(3)
        rng = rng.advance(7)
        assert rng.consumed == 10


# ---------------------------------------------------------------------------
# AbstractMCMCTrace
# ---------------------------------------------------------------------------


class TestAbstractMCMCTrace:
    def test_create(self):
        t = AbstractMCMCTrace(param_dims=(3,), warmup_steps=100)
        assert t.n_samples == 0
        assert t.n_chains == 1
        assert not t.is_warmed_up

    def test_step(self):
        t = AbstractMCMCTrace(param_dims=(3,), warmup_steps=2)
        t2 = t.step(accepted=True)
        assert t2.n_samples == 1
        assert not t2.is_warmed_up

    def test_warmup_completes(self):
        t = AbstractMCMCTrace(param_dims=(3,), warmup_steps=2)
        t = t.step()
        assert not t.is_warmed_up
        t = t.step()
        assert t.is_warmed_up

    def test_assert_warmed_up_fail(self):
        t = AbstractMCMCTrace(param_dims=(3,), warmup_steps=100)
        with pytest.raises(ValueError, match="not warmed up"):
            t.assert_warmed_up()

    def test_assert_warmed_up_pass(self):
        t = AbstractMCMCTrace(
            param_dims=(3,), warmup_steps=0, n_samples=1, is_warmed_up=True
        )
        t.assert_warmed_up()

    def test_assert_param_dims(self):
        t = AbstractMCMCTrace(param_dims=(3, 2))
        t.assert_param_dims((3, 2))
        with pytest.raises(ValueError, match="dimension mismatch"):
            t.assert_param_dims((5,))

    def test_accept_rate_tracking(self):
        t = AbstractMCMCTrace(param_dims=(1,), warmup_steps=0)
        # First accepted step
        t = t.step(accepted=True)
        assert t.accept_rate > 0
        # Rejected step should lower rate
        prev_rate = t.accept_rate
        t = t.step(accepted=False)
        assert t.accept_rate < prev_rate

    def test_multi_chain(self):
        t = AbstractMCMCTrace(param_dims=(3,), n_chains=4)
        assert t.n_chains == 4


# ---------------------------------------------------------------------------
# Conjugate pairs reference
# ---------------------------------------------------------------------------


class TestConjugatePairs:
    def test_known_pairs_exist(self):
        assert ("bernoulli", "beta") in CONJUGATE_PAIRS
        assert ("categorical", "dirichlet") in CONJUGATE_PAIRS
        assert ("poisson", "gamma") in CONJUGATE_PAIRS
        assert ("normal", "normal") in CONJUGATE_PAIRS

    def test_distribution_families(self):
        assert "normal" in DISTRIBUTION_FAMILIES
        assert "dirichlet" in DISTRIBUTION_FAMILIES
        assert "wishart" in DISTRIBUTION_FAMILIES


# ---------------------------------------------------------------------------
# witness_prior_init
# ---------------------------------------------------------------------------


class TestWitnessPriorInit:
    def test_normal_prior(self):
        d = witness_prior_init(event_shape=(3,), family="normal")
        assert d.family == "normal"
        assert d.event_shape == (3,)
        assert not d.is_discrete

    def test_beta_prior(self):
        d = witness_prior_init(event_shape=(1,), family="beta")
        assert d.support_lower == 0.0
        assert d.support_upper == 1.0

    def test_gamma_prior(self):
        d = witness_prior_init(event_shape=(1,), family="gamma")
        assert d.support_lower == 0.0

    def test_categorical_prior(self):
        d = witness_prior_init(event_shape=(5,), family="categorical")
        assert d.is_discrete

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution family"):
            witness_prior_init(event_shape=(1,), family="invented")

    def test_empty_shape_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            witness_prior_init(event_shape=(), family="normal")


# ---------------------------------------------------------------------------
# witness_log_prob
# ---------------------------------------------------------------------------


class TestWitnessLogProb:
    def test_compatible_shapes(self):
        dist = AbstractDistribution(family="normal", event_shape=(3,))
        samples = AbstractArray(shape=(100, 3))
        result = witness_log_prob(dist, samples)
        assert result.dtype == "float64"
        assert result.max_val == 0.0

    def test_incompatible_shapes_raises(self):
        dist = AbstractDistribution(family="normal", event_shape=(3,))
        samples = AbstractArray(shape=(100, 5))
        with pytest.raises(ValueError, match="don't match"):
            witness_log_prob(dist, samples)

    def test_scalar_event(self):
        dist = AbstractDistribution(family="normal", event_shape=(1,))
        samples = AbstractArray(shape=(50, 1))
        result = witness_log_prob(dist, samples)
        assert isinstance(result, AbstractScalar)


# ---------------------------------------------------------------------------
# witness_mcmc_step
# ---------------------------------------------------------------------------


class TestWitnessMcmcStep:
    def test_basic_step(self):
        trace = AbstractMCMCTrace(param_dims=(3,), warmup_steps=10)
        target = AbstractDistribution(family="normal", event_shape=(3,))
        rng = AbstractRNGState(seed=42)

        new_trace, new_rng = witness_mcmc_step(trace, target, rng)
        assert new_trace.n_samples == 1
        assert new_rng.consumed == 1

    def test_dim_mismatch_raises(self):
        trace = AbstractMCMCTrace(param_dims=(3,))
        target = AbstractDistribution(family="normal", event_shape=(5,))
        rng = AbstractRNGState(seed=42)

        with pytest.raises(ValueError, match="parameter dims"):
            witness_mcmc_step(trace, target, rng)

    def test_negative_step_size_raises(self):
        trace = AbstractMCMCTrace(param_dims=(3,))
        target = AbstractDistribution(family="normal", event_shape=(3,))
        rng = AbstractRNGState(seed=42)

        with pytest.raises(ValueError, match="step_size must be positive"):
            witness_mcmc_step(trace, target, rng, step_size=-0.1)

    def test_rng_advances(self):
        trace = AbstractMCMCTrace(param_dims=(2,), warmup_steps=0)
        target = AbstractDistribution(family="normal", event_shape=(2,))
        rng = AbstractRNGState(seed=0)

        # Multiple steps
        for i in range(5):
            trace, rng = witness_mcmc_step(trace, target, rng)

        assert trace.n_samples == 5
        assert rng.consumed == 5
        assert trace.is_warmed_up

    def test_warmup_transition(self):
        trace = AbstractMCMCTrace(param_dims=(2,), warmup_steps=3)
        target = AbstractDistribution(family="normal", event_shape=(2,))
        rng = AbstractRNGState(seed=0)

        for _ in range(2):
            trace, rng = witness_mcmc_step(trace, target, rng)
        assert not trace.is_warmed_up

        trace, rng = witness_mcmc_step(trace, target, rng)
        assert trace.is_warmed_up


# ---------------------------------------------------------------------------
# witness_posterior_update
# ---------------------------------------------------------------------------


class TestWitnessPosteriorUpdate:
    def test_beta_bernoulli(self):
        prior = AbstractDistribution(
            family="beta", event_shape=(1,),
            support_lower=0.0, support_upper=1.0,
        )
        likelihood = AbstractDistribution(
            family="bernoulli", event_shape=(1,), is_discrete=True,
        )
        posterior = witness_posterior_update(prior, likelihood, data_shape=(100, 1))
        assert posterior.family == "beta"  # conjugate closure
        assert posterior.event_shape == (1,)
        assert posterior.support_lower == 0.0

    def test_dirichlet_categorical(self):
        prior = AbstractDistribution(family="dirichlet", event_shape=(5,))
        likelihood = AbstractDistribution(
            family="categorical", event_shape=(5,), is_discrete=True,
        )
        posterior = witness_posterior_update(prior, likelihood, data_shape=(200, 5))
        assert posterior.family == "dirichlet"

    def test_non_conjugate_raises(self):
        prior = AbstractDistribution(family="beta", event_shape=(1,))
        likelihood = AbstractDistribution(family="normal", event_shape=(1,))
        with pytest.raises(ValueError, match="not a conjugate prior"):
            witness_posterior_update(prior, likelihood, data_shape=(100, 1))

    def test_normal_normal(self):
        prior = AbstractDistribution(family="normal", event_shape=(3,))
        likelihood = AbstractDistribution(family="normal", event_shape=(3,))
        posterior = witness_posterior_update(prior, likelihood, data_shape=(50, 3))
        assert posterior.family == "normal"
        assert posterior.event_shape == (3,)


# ---------------------------------------------------------------------------
# witness_vi_elbo
# ---------------------------------------------------------------------------


class TestWitnessViElbo:
    def test_compatible(self):
        q = AbstractDistribution(family="normal", event_shape=(3,))
        p = AbstractDistribution(family="normal", event_shape=(3,))
        result = witness_vi_elbo(q, p, n_samples=10)
        assert result.dtype == "float64"

    def test_shape_mismatch_raises(self):
        q = AbstractDistribution(family="normal", event_shape=(3,))
        p = AbstractDistribution(family="normal", event_shape=(5,))
        with pytest.raises(ValueError, match="event_shape"):
            witness_vi_elbo(q, p)

    def test_zero_samples_raises(self):
        q = AbstractDistribution(family="normal", event_shape=(3,))
        p = AbstractDistribution(family="normal", event_shape=(3,))
        with pytest.raises(ValueError, match="n_samples must be positive"):
            witness_vi_elbo(q, p, n_samples=0)
