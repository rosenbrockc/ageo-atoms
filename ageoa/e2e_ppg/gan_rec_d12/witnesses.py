def witness_normalize_and_batch_clean_segment(
    clean_segment: AbstractArray,
    sampling_rate: AbstractScalar,
) -> AbstractArray:
    """Ghost witness for normalize_and_batch_clean_segment.

    The real function normalises the raw segment and wraps it in a leading
    batch dimension so that downstream vectorised operations work correctly.
    The witness must mirror that shape contract: (1, *segment_shape).  The
    original witness incorrectly returned `shape=clean_segment.shape` (no
    batch axis), which made the abstract graph see a shape-preserving
    identity at this node and caused the simulator to detect a cycle among
    {'normalize_and_batch_clean_segment',
     'stitch_clean_and_reconstructed_waveforms',
     'generate_reconstructed_segment',
     'detect_peaks_in_reconstructed_and_clean',
     'accumulate_reconstructed_noise_and_advance_window'}.
    Prepending the batch dimension breaks that cycle.
    """
    result = AbstractArray(
        shape=(1,) + clean_segment.shape,
        dtype="float64",
    )
    return result