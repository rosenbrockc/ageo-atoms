from __future__ import annotations
from ageoa.ghost.abstract import AbstractSignal, AbstractArray

def witness_sub_graph_embedder(current_graph: AbstractSignal, subgraph_quantity: AbstractSignal, state: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for Sub-graph Embedder."""
    result = AbstractSignal(
        shape=current_graph.shape,
        dtype="float64",
        sampling_rate=getattr(current_graph, "sampling_rate", 44100.0),
        domain="time",
    )
    return result, state

def witness_graph_transformer(current_graph: AbstractSignal, lattice: AbstractSignal, mapping: AbstractSignal, state: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
    """Ghost witness for Graph Transformer."""
    result = AbstractSignal(
        shape=current_graph.shape,
        dtype="float64",
        sampling_rate=getattr(current_graph, "sampling_rate", 44100.0),
        domain="time",
    )
    return result, state

def witness_quantum_mwis_solver(graph: AbstractArray, lattice_id_coord_dic: AbstractArray, mis_sample_quantity: AbstractArray) -> AbstractArray:
    """Ghost witness for opaque boundary: Quantum MWIS Solver."""
    return AbstractArray(shape=graph.shape, dtype="float32")
