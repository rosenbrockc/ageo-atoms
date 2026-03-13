from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_load_graphs_from_folder(folder_path, *args, **kwargs):
    """Ghost witness for Load Graphs From Folder."""
    result = AbstractArray(
        shape=(1,),
        dtype="float64",)

    return result

def witness_is_independent_set(graph: AbstractArray, subset: AbstractArray) -> AbstractArray:
    """Ghost witness for Is Independent Set."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result

def witness_calculate_weight(graph: AbstractArray, node_list: AbstractArray) -> AbstractArray:
    """Ghost witness for Calculate Weight."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result

def witness_to_qubo(graph: AbstractArray, penalty: AbstractArray) -> AbstractArray:
    """Ghost witness for To Qubo."""
    result = AbstractArray(
        shape=graph.shape,
        dtype="float64",)

    return result
