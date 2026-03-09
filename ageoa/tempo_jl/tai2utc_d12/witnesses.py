def witness_utc_to_tai_leap_second_kernel(
    utc1: AbstractArray,
    utc2: AbstractArray,
) -> AbstractScalar:
    """Ghost witness for utc_to_tai_leap_second_kernel.

    Pins both the type *and* the value-domain of the returned leap-second
    offset to a fully concrete, 0-D scalar.  Setting ``values=ANYTHING``
    provides the simulator with a grounded value sentinel so it never needs
    to chase a symbolic back-edge into tai_to_utc_inversion when widening
    the return type, which was the root cause of the detected cycle.
    """
    result = AbstractScalar(
        dtype="float64",
        shape=(),
        ndim=0,
        values=ANYTHING,  # concrete sentinel — severs symbolic link to inversion path
    )
    return result
