def witness_predictlatentstateandcovariance(state_in: AbstractArray, u: AbstractArray, B: AbstractArray, F: AbstractArray, Q: AbstractArray) -> AbstractArray:
    result = AbstractArray(
        shape=state_in.shape,
        dtype="float64",
    )
    return result
