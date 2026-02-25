_stage_order = {
    'runsamplingloop': 0,
    'metropolishmctransition': 1,
    'preparedifferentiablestateandfirsthalfkick': 2,
    'leapfrogproposalkernel': 3,
    'recomputegradienthalfkickcontribution': 4,
    'driftpositionbyfullstep': 5,
    'completemomentumkickandrefreshcarry': 6,
    'computeterminallogprobandemitproposal': 7,
}

edge_definitions = [
    e
    for e in edge_definitions
    if not (
        e.source_id in _stage_order
        and e.target_id in _stage_order
        and _stage_order[e.source_id] >= _stage_order[e.target_id]
    )
]
