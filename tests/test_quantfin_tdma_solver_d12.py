from ageoa.quantfin.tdma_solver_d12.atoms import cotraversevec


def test_cotraversevec_aggregates_projected_vector_entries() -> None:
    result = cotraversevec(
        lambda start, length: list(range(start, length)),
        lambda projected: float(sum(projected)),
        lambda projector, wrapped: [projector(vec) for vec in wrapped],
        0,
        3,
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
        lambda mapper, indices: [mapper(idx) for idx in indices],
    )

    assert result == [11.0, 22.0, 33.0]
