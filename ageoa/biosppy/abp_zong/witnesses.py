def advance_or_terminate_scan(
    signal: AbstractSignal,
    accepted_onsets: AbstractSignal,
    size: AbstractSignal,
) -> AbstractSignal:
    return AbstractSignal(
        shape=(1,),
        dtype="bool",
        sampling_rate=getattr(signal, "sampling_rate", 44100.0),
        domain="message",
    )
