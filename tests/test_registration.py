"""Tests verifying that @register_atom wiring is correct."""

import pytest

from ageoa.ghost.registry import REGISTRY, list_registered, get_witness


# Force-import all atom modules so decorators fire
import ageoa.numpy.fft
import ageoa.scipy.fft
import ageoa.scipy.signal
import ageoa.scipy.sparse_graph
import ageoa.rust_robotics
import ageoa.tempo
import ageoa.quantfin


EXPECTED_ATOMS = [
    "fft", "ifft", "rfft", "irfft",
    "dct", "idct",
    "butter", "cheby1", "cheby2", "firwin",
    "sosfilt", "lfilter", "freqz",
    "graph_laplacian", "graph_fourier_transform", "heat_kernel_diffusion",
    "pure_pursuit", "rk4",
    "offset_tt2tdb", "offset_tai2tdb",
    "run_simulation_anti", "quick_sim_anti",
]


class TestRegistration:
    """Verify that all heavy atoms are registered with ghost witnesses."""

    def test_all_16_atoms_registered(self):
        registered = list_registered()
        assert len(registered) >= len(EXPECTED_ATOMS)

    @pytest.mark.parametrize("atom_name", EXPECTED_ATOMS)
    def test_atom_in_registry(self, atom_name):
        assert atom_name in REGISTRY

    @pytest.mark.parametrize("atom_name", EXPECTED_ATOMS)
    def test_atom_has_witness(self, atom_name):
        witness = get_witness(atom_name)
        assert callable(witness)

    @pytest.mark.parametrize("atom_name", EXPECTED_ATOMS)
    def test_atom_has_impl(self, atom_name):
        entry = REGISTRY[atom_name]
        assert "impl" in entry
        assert callable(entry["impl"])

    @pytest.mark.parametrize("atom_name", EXPECTED_ATOMS)
    def test_atom_has_signature(self, atom_name):
        entry = REGISTRY[atom_name]
        assert "signature" in entry
        assert isinstance(entry["signature"], dict)
        # Witness must have a return annotation
        assert "return" in entry["signature"]

    @pytest.mark.parametrize("atom_name", EXPECTED_ATOMS)
    def test_atom_has_module(self, atom_name):
        entry = REGISTRY[atom_name]
        assert "module" in entry
        assert entry["module"]  # non-empty


class TestRegistryLookupErrors:
    """Verify that get_witness raises KeyError for unknown atoms."""

    def test_unknown_atom_raises_keyerror(self):
        with pytest.raises(KeyError, match="No ghost witness"):
            get_witness("nonexistent_atom_xyz")


class TestImplMatchesHeavy:
    """Verify that the registered impl is the actual heavy function."""

    def test_fft_impl_is_the_heavy_function(self):
        entry = REGISTRY["fft"]
        # The impl should be the icontract-wrapped heavy function
        assert entry["impl"].__name__ == "fft"

    def test_butter_impl_is_the_heavy_function(self):
        entry = REGISTRY["butter"]
        assert entry["impl"].__name__ == "butter"

    def test_graph_laplacian_impl_is_the_heavy_function(self):
        entry = REGISTRY["graph_laplacian"]
        assert entry["impl"].__name__ == "graph_laplacian"
