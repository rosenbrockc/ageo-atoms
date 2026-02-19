import jax.numpy as jnp
import pytest
from ageoa.alphafold.atoms import (
    invariant_point_attention,
    equivariant_frame_update,
    coordinate_reconstruction
)
from ageoa.alphafold.state_models import AlphaFoldStructuralState

def test_alphafold_structural_pipeline():
    n_res = 10
    c_n = 256
    c_p = 128
    
    nodes = jnp.zeros((n_res, c_n))
    pairs = jnp.zeros((n_res, n_res, c_p))
    frames = jnp.zeros((n_res, 7)) # Simplified rigid representation
    torsions = jnp.zeros((n_res, 7, 2))
    
    state = AlphaFoldStructuralState()
    
    # 1. IPA
    nodes_up, state = invariant_point_attention(nodes, pairs, frames, state)
    assert nodes_up.shape == (n_res, c_n)
    
    # 2. Frame update
    frames_up, state = equivariant_frame_update(frames, nodes_up, state)
    assert frames_up.shape == (n_res, 7)
    
    # 3. Coordinate reconstruction
    coords, state = coordinate_reconstruction(frames_up, torsions, state)
    assert coords.shape == (n_res, 37, 3)

def test_ipa_mismatch():
    n_res = 10
    nodes = jnp.zeros((n_res, 256))
    pairs = jnp.zeros((n_res + 1, n_res, 128)) # Mismatch
    frames = jnp.zeros((n_res, 7))
    state = AlphaFoldStructuralState()
    
    with pytest.raises(Exception):
        invariant_point_attention(nodes, pairs, frames, state)
