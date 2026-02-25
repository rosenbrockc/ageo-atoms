def repair_edge_definitions(edge_definitions):
    """Break non-message-passing reconstruction loop by removing feedback edge."""

    def _read(edge, *keys):
        for k in keys:
            if isinstance(edge, dict) and k in edge:
                return edge[k]
            v = getattr(edge, k, None)
            if v is not None:
                return v
        return None

    def _canon(node_id):
        # Match IDs robustly across snake_case/camelCase/spaces.
        return "".join(ch for ch in str(node_id or "").lower() if ch.isalnum())

    blocked_edges = {
        ("checkreconstructioncompletion", "buildalignedreconstructioncandidate"),
    }

    repaired = []
    for e in edge_definitions:
        src = _canon(_read(e, "source_id", "source", "from_id", "from"))
        dst = _canon(_read(e, "target_id", "target", "to_id", "to"))
        if (src, dst) in blocked_edges:
            continue
        repaired.append(e)

    return repaired
