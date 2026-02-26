"""Ghost witnesses for mini_mcmc HMC atoms."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractArray, AbstractMCMCTrace, AbstractRNGState, AbstractScalar
except ImportError:
    pass


def witness_initializehmcstate(
    target: AbstractArray,
    initial_positions: AbstractScalar,
    step_size: AbstractScalar,
    n_leapfrog: AbstractScalar,
    seed: AbstractScalar,
) -> tuple[AbstractMCMCTrace, AbstractArray]:
    """Ghost witness for InitializeHMCState.

    Args:
        target: Density evaluator metadata.
        initial_positions: Starting position metadata.
        step_size: Leapfrog step size metadata.
        n_leapfrog: Number of leapfrog steps.
        seed: RNG seed metadata.

    Returns:
        Tuple of (HMC state trace, kernel static config).
    """
    hmc_state = AbstractMCMCTrace(n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0)
    kernel_static = AbstractArray(shape=("K",), dtype="float64")
    return (hmc_state, kernel_static)


def witness_leapfrogproposalkernel(
    proposal_state_in: AbstractMCMCTrace,
    kernel_static: AbstractArray,
    log_prob_oracle: AbstractArray,
) -> AbstractMCMCTrace:
    """Ghost witness for LeapfrogProposalKernel.

    Args:
        proposal_state_in: Current proposal state metadata.
        kernel_static: Kernel configuration metadata.
        log_prob_oracle: Log-probability evaluator metadata.

    Returns:
        Updated proposal state metadata.
    """
    return AbstractMCMCTrace(
        n_samples=proposal_state_in.n_samples,
        n_chains=proposal_state_in.n_chains,
        param_dims=proposal_state_in.param_dims,
        warmup_steps=proposal_state_in.warmup_steps,
    )


def witness_metropolishmctransition(
    chain_state_in: AbstractMCMCTrace,
    kernel_static: AbstractArray,
    proposal_state_out: AbstractMCMCTrace,
) -> tuple[AbstractMCMCTrace, AbstractArray]:
    """Ghost witness for MetropolisHMCTransition.

    Args:
        chain_state_in: Current chain state metadata.
        kernel_static: Kernel configuration metadata.
        proposal_state_out: Proposal from leapfrog metadata.

    Returns:
        Tuple of (updated chain state, transition statistics).
    """
    chain_state_out = AbstractMCMCTrace(
        n_samples=chain_state_in.n_samples,
        n_chains=chain_state_in.n_chains,
        param_dims=chain_state_in.param_dims,
        warmup_steps=chain_state_in.warmup_steps,
    )
    stats = AbstractArray(shape=("3",), dtype="float64")
    return (chain_state_out, stats)


def witness_runsamplingloop(
    hmc_state_in: AbstractMCMCTrace,
    n_collect: AbstractScalar,
    n_discard: AbstractScalar,
) -> tuple[AbstractArray, AbstractMCMCTrace, AbstractMCMCTrace]:
    """Ghost witness for RunSamplingLoop.

    Args:
        hmc_state_in: Initial HMC state metadata.
        n_collect: Number of collection samples.
        n_discard: Number of warmup samples to discard.

    Returns:
        Tuple of (collected samples, trace diagnostics, final HMC state).
    """
    samples = AbstractArray(shape=("N", "D"), dtype="float64")
    trace = AbstractMCMCTrace(
        n_samples=0,
        n_chains=hmc_state_in.n_chains,
        param_dims=hmc_state_in.param_dims,
        warmup_steps=0,
    )
    final_state = AbstractMCMCTrace(
        n_samples=hmc_state_in.n_samples,
        n_chains=hmc_state_in.n_chains,
        param_dims=hmc_state_in.param_dims,
        warmup_steps=hmc_state_in.warmup_steps,
    )
    return (samples, trace, final_state)
