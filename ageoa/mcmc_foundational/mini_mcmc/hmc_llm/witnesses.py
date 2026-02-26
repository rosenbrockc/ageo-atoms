"""Ghost witnesses for HMC LLM atoms."""

from __future__ import annotations

try:
    from ageoa.ghost.abstract import AbstractArray, AbstractMCMCTrace, AbstractRNGState, AbstractScalar
except ImportError:
    pass


def witness_initializehmckernelstate(
    target: AbstractArray,
    initial_positions: AbstractArray,
    step_size: AbstractScalar,
    n_leapfrog: AbstractScalar,
) -> tuple[AbstractMCMCTrace, AbstractArray]:
    """Ghost witness for InitializeHMCKernelState.

    Args:
        target: Density evaluator metadata.
        initial_positions: Starting position metadata.
        step_size: Leapfrog step size metadata.
        n_leapfrog: Number of leapfrog steps.

    Returns:
        Tuple of (kernel spec, initial chain state).
    """
    kernel_spec = AbstractArray(shape=("K",), dtype="float64")
    chain_state = AbstractMCMCTrace(n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0)
    return (chain_state, kernel_spec)


def witness_initializesamplerrng(
    seed: AbstractScalar,
) -> AbstractRNGState:
    """Ghost witness for InitializeSamplerRNG.

    Args:
        seed: RNG seed metadata.

    Returns:
        Initialized RNG state.
    """
    return AbstractRNGState(seed=0, consumed=0, is_split=False)


def witness_hamiltoniantransitionkernel(
    state_in: AbstractMCMCTrace,
    kernel_spec: AbstractArray,
    prng_key_in: AbstractRNGState,
    logp_oracle: AbstractArray,
) -> tuple[AbstractMCMCTrace, AbstractRNGState, AbstractArray]:
    """Ghost witness for HamiltonianTransitionKernel.

    Args:
        state_in: Current chain state metadata.
        kernel_spec: Kernel configuration metadata.
        prng_key_in: Current RNG state.
        logp_oracle: Log-probability evaluator metadata.

    Returns:
        Tuple of (updated state, new RNG key, transition stats).
    """
    state_out = AbstractMCMCTrace(
        n_samples=state_in.n_samples,
        n_chains=state_in.n_chains,
        param_dims=state_in.param_dims,
        warmup_steps=state_in.warmup_steps,
    )
    rng_out = AbstractRNGState(seed=0, consumed=1, is_split=True)
    stats = AbstractArray(shape=("3",), dtype="float64")
    return (state_out, rng_out, stats)


def witness_collectposteriorchain(
    n_collect: AbstractScalar,
    n_discard: AbstractScalar,
    chain_state_0: AbstractMCMCTrace,
    kernel_spec: AbstractArray,
    prng_key_state: AbstractRNGState,
) -> tuple[AbstractArray, AbstractMCMCTrace, AbstractRNGState, AbstractMCMCTrace]:
    """Ghost witness for CollectPosteriorChain.

    Args:
        n_collect: Number of collection samples.
        n_discard: Number of warmup samples.
        chain_state_0: Initial chain state.
        kernel_spec: Kernel configuration.
        prng_key_state: RNG state.

    Returns:
        Tuple of (samples, final state, final RNG, chain trace).
    """
    samples = AbstractArray(shape=("N", "D"), dtype="float64")
    final_state = AbstractMCMCTrace(
        n_samples=chain_state_0.n_samples,
        n_chains=chain_state_0.n_chains,
        param_dims=chain_state_0.param_dims,
        warmup_steps=chain_state_0.warmup_steps,
    )
    final_rng = AbstractRNGState(seed=0, consumed=0, is_split=False)
    trace = AbstractMCMCTrace(
        n_samples=0, n_chains=1, param_dims=(1,), warmup_steps=0,
    )
    return (samples, final_state, final_rng, trace)
