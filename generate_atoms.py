import os
from pathlib import Path

REPOS = {
    "e2e_ppg": {
        "atoms": ["kazemi_peak_detection", "ppg_reconstruction", "ppg_sqa"],
        "desc": [
            "Extracts local maxima from a wandering 1D scalar signal array.",
            "Reconstructs corrupted segments of a 1D scalar sequence.",
            "Quantifies the reliability and signal-to-noise ratio of a 1D scalar array."
        ]
    },
    "institutional_quant_engine": {
        "atoms": ["market_making_avellaneda", "almgren_chriss_execution", "pin_informed_trading", "limit_order_queue_estimator"],
        "desc": [
            "Adjusts boundary thresholds based on inventory states and variance.",
            "Calculates the optimal trajectory for liquidating a large state variable.",
            "Estimates the probability of asymmetric information from sequence flow data.",
            "Estimates the discrete position of an item within a prioritized queue."
        ]
    },
    "quantfin": {
        "atoms": ["functional_monte_carlo", "volatility_surface_modeling"],
        "desc": [
            "Generates stochastic paths and evaluates contingent claims using functional constraints.",
            "Interpolates and calibrates an implied variance surface."
        ]
    },
    "pulsar_folding": {
        "atoms": ["dm_can_brute_force", "spline_bandpass_correction"],
        "desc": [
            "Performs a brute-force shift search to maximize the signal-to-noise ratio of a folded profile.",
            "Subtracts instrument-induced artifacts across frequency channels using interpolative splines."
        ]
    },
    "tempo_jl": {
        "atoms": ["graph_time_scale_management", "high_precision_duration"],
        "desc": [
            "Computes transformation paths dynamically using a directed graph representation.",
            "Splits a continuous variable into an integer and fractional part to preserve numerical precision."
        ]
    },
    "pronto": {
        "atoms": ["rbis_state_estimation"],
        "desc": [
            "Provides a recursive Bayesian incremental state estimation framework for sensor fusion."
        ]
    },
    "rust_robotics": {
        "atoms": ["n_joint_arm_solver", "dijkstra_path_planning"],
        "desc": [
            "Solves custom kinematics and dynamics for an N-joint system.",
            "Computes the shortest path on a weighted graph from a single source node."
        ]
    },
    "molecular_docking": {
        "atoms": ["quantum_mwis_solver", "greedy_lattice_mapping"],
        "desc": [
            "Solves the Maximum Weight Independent Set problem on a graph using quantum heuristics.",
            "Maps abstract interaction graphs onto physical 2D lattices under hardware constraints."
        ]
    },
    "mint": {
        "atoms": ["axial_attention", "rotary_positional_embeddings"],
        "desc": [
            "Implements factorized attention over 2D sequence alignments.",
            "Encodes relative position into a tensor using rotary transformations."
        ]
    }
}

ATOM_TEMPLATE = """@register_atom(witness_{atom})
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def {atom}(data: np.ndarray) -> np.ndarray:
    \"\"\"{desc}

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    \"\"\"
    raise NotImplementedError("Skeleton for future ingestion.")

"""

WITNESS_TEMPLATE = """def witness_{atom}(data: AbstractArray) -> AbstractArray:
    \"\"\"Witness for {atom}.\"\"\"
    return AbstractArray(shape=data.shape, dtype=data.dtype)

"""

TEST_TEMPLATE = """def test_{atom}_positive():
    with pytest.raises(NotImplementedError):
        {atom}(np.array([1.0]))

def test_{atom}_precondition():
    with pytest.raises(icontract.ViolationError):
        {atom}(None)

"""

def main():
    for repo, data in REPOS.items():
        base = Path("ageoa") / repo
        base.mkdir(parents=True, exist_ok=True)
        
        atoms_py = base / "atoms.py"
        witnesses_py = base / "witnesses.py"
        init_py = base / "__init__.py"
        
        test_file = Path("tests") / f"test_{repo}.py"
        
        atoms_code = '"""Auto-generated verified atom wrapper."""\\n\\nimport numpy as np\\nimport icontract\\nfrom ageoa.ghost.registry import register_atom\\n'
        witnesses_code = '"""Ghost witnesses."""\\n\\nfrom ageoa.ghost.abstract import AbstractArray\\n\\n'
        test_code = f'"""Tests for {repo}."""\\n\\nimport pytest\\nimport numpy as np\\nimport icontract\\n'
        
        exports = []
        for i, atom in enumerate(data["atoms"]):
            desc = data["desc"][i] if isinstance(data["desc"], list) else data["desc"]
            
            atoms_code += f"from ageoa.{repo}.witnesses import witness_{atom}\\n"
            test_code += f"from ageoa.{repo}.atoms import {atom}\\n"
            
            exports.append(atom)
            
        atoms_code += "\\n"
        test_code += "\\n"

        for i, atom in enumerate(data["atoms"]):
            desc = data["desc"][i] if isinstance(data["desc"], list) else data["desc"]
            atoms_code += ATOM_TEMPLATE.format(atom=atom, desc=desc)
            witnesses_code += WITNESS_TEMPLATE.format(atom=atom)
            test_code += TEST_TEMPLATE.format(atom=atom)
            
        atoms_py.write_text(atoms_code)
        witnesses_py.write_text(witnesses_code)
        init_py.write_text(f"from .atoms import {', '.join(exports)}\\n\\n__all__ = {repr(exports)}\\n")
        test_file.write_text(test_code)
        print(f"Generated {repo}")

if __name__ == "__main__":
    main()
