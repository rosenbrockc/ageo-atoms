"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

witness_graphtoudgmapping: object = object()
@register_atom(witness_graphtoudgmapping)  # type: ignore[untyped-decorator]
@register_atom(witness_graphtoudgmapping)  # type: ignore[untyped-decorator]
@icontract.ensure(lambda result, **kwargs: result is not None, "GraphToUDGMapping output must not be None")
def graphtoudgmapping(G: nx.Graph) -> nx.Graph:
    """Map a graph to a UDG mapping.

    Args:
        G: Must be a valid graph object accepted by map_to_UDG.

    Returns:
        New mapped graph output; no hidden state mutation.
    """
    raise NotImplementedError("Wire to original implementation")