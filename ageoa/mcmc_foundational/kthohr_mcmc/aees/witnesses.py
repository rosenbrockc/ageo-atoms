def witness_metropolishastingstransitionkernel(temper_val: AbstractArray, target_log_kernel: object, rng_key_in: AbstractArray) -> AbstractSignal:
    """Ghost witness for MetropolisHastingsTransitionKernel.

    `target_log_kernel` is a static oracle descriptor, not a streamed signal.
    """
    result = AbstractSignal(
        shape=temper_val.shape,
        dtype="float64",
        sampling_rate=1.0,
        domain="iteration",
        units="state",
    )
    return result
