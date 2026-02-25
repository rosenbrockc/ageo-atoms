def witness_extractsegmentminimumintobuffers(*args, **kwargs):
    """Cycle-breaking witness for ExtractSegmentMinimumIntoBuffers."""
    minima = kwargs.get("buffered_minima")
    if minima is not None:
        if isinstance(minima, dict) and "minima" in minima:
            vals = minima.get("minima") or ()
            return {"minima": tuple(vals), "count": len(vals)}
        if isinstance(minima, (list, tuple)):
            return {"minima": tuple(minima), "count": len(minima)}

    for arg in args:
        if isinstance(arg, dict) and "minima" in arg:
            vals = arg.get("minima") or ()
            return {"minima": tuple(vals), "count": len(vals)}
        if isinstance(arg, (list, tuple)):
            return {"minima": tuple(arg), "count": len(arg)}

    return {"minima": (), "count": 0}
