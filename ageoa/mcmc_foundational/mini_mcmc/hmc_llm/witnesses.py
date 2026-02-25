def witness_hamiltoniantransitionkernel(state_in: AbstractArray, kernel_spec: AbstractArray, prng_key_in: AbstractArray, logp_oracle: AbstractArray) -> AbstractArray:
    """Ghost witness for HamiltonianTransitionKernel."""
    # Preserve the transition dependency by propagating incoming state.
    return state_in
