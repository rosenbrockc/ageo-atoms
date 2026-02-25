def witness_meanfieldvariationalfit(
    q_dist: AbstractDistribution | None = None,
    p_dist: AbstractDistribution | None = None,
    n_samples: int = 1,
) -> AbstractDistribution:
    """Ghost witness for VI fit: MeanFieldVariationalFit returns a posterior-like distribution."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if q_dist is not None and p_dist is not None and q_dist.event_shape != p_dist.event_shape:
        raise ValueError(
            f"q event_shape {q_dist.event_shape} vs p event_shape {p_dist.event_shape}"
        )

    if q_dist is not None:
        return q_dist
    if p_dist is not None:
        return p_dist
    raise ValueError("At least one of q_dist or p_dist must be provided")
