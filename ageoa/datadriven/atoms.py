"""DataDrivenDiffEq SINDy Macro-Atoms."""

import numpy as np
import icontract
from juliacall import Main as jl
from pydantic import BaseModel, Field
from typing import List, Optional
from ageoa.ghost.registry import register_atom
from ageoa.datadriven.witnesses import witness_discover_equations

jl.seval("using DataDrivenDiffEq")
jl.seval("using DataDrivenSparse")
jl.seval("using ModelingToolkit")

class EquationResult(BaseModel):
    """Result container for symbolic equation discovery."""
    equations: List[str] = Field(description="The discovered symbolic equations.")
    parameter_map: dict = Field(description="Map of parameter values discovered.")

@register_atom(witness_discover_equations)
@icontract.require(lambda X: X.ndim == 2, "X must be a 2D array (features x samples)")
@icontract.require(lambda Y: Y.ndim == 2, "Y must be a 2D array (targets x samples)")
@icontract.require(lambda X, Y: X.shape[1] == Y.shape[1], "X and Y must have same number of samples")
@icontract.require(lambda lambda_val: lambda_val > 0, "Sparsity penalty lambda_val must be positive")
def discover_equations(
    X: np.ndarray,
    Y: np.ndarray,
    variable_names: List[str],
    max_degree: int = 4,
    lambda_val: float = 0.1
) -> EquationResult:
    """Sparsity-promoting symbolic regression for discovering governing equations from data.

    Applies sparsity-promoting regression to identify a minimal set of symbolic
    terms that govern the dynamics of the input measurements.

    Args:
        X: The system state measurements (features x samples)
        Y: The target derivatives or function values (targets x samples)
        variable_names: A list of strings defining the symbolic identifiers for each state variable.
        max_degree: Maximum polynomial basis degree.
        lambda_val: Sparsity penalty (Occam's razor heuristic parameter).

    Returns:
        EquationResult with the parsed discovered equations and parameter values.
    """
    if X.shape[0] != len(variable_names):
        raise ValueError(f"X feature dimension {X.shape[0]} does not match number of variable names {len(variable_names)}")
    
    # Declarative metadata injected into Julia MTK.@variables macro dynamically
    var_str = " ".join(variable_names)
    jl.seval(f"ModelingToolkit.@variables {var_str}")
    
    # Pack into an array of symbolic variables
    jl.seval(f"u_vars = [{', '.join(variable_names)}]")
    
    # Build polynomial basis
    jl.seval(f"b = DataDrivenDiffEq.polynomial_basis(u_vars, {max_degree})")
    basis = jl.seval("DataDrivenDiffEq.Basis(b, u_vars)")
    
    # Copy data securely over FFI
    jl.X_train = X.astype(np.float64)
    jl.Y_train = Y.astype(np.float64)
    
    # Setup DataDrivenProblem
    prob = jl.seval("DataDrivenDiffEq.DirectDataDrivenProblem(X_train, Y_train)")
    
    # Apply sparsity-promoting heuristic (Occam's razor via ADMM)
    jl.lambda_val = float(lambda_val)
    opt = jl.seval("DataDrivenSparse.ADMM(lambda_val)")
    
    # Solve regression
    jl.prob = prob
    jl.basis = basis
    jl.opt = opt
    res = jl.seval("DataDrivenDiffEq.solve(prob, basis, opt)")
    jl.res = res
    
    # Extract basis and parameter mapping
    basis_res = jl.seval("DataDrivenDiffEq.get_basis(res)")
    jl.basis_res = basis_res
    eqs_str = str(basis_res)
    eqs = [line.strip() for line in eqs_str.split(chr(10)) if line.strip()]
    
    param_map = jl.seval("DataDrivenDiffEq.get_parameter_map(basis_res)")
    pm_dict = {}
    for p_tuple in param_map:
        # p_tuple is a Pair (Symbolic -> Float64)
        sym = str(p_tuple[0])
        val = float(p_tuple[1])
        pm_dict[sym] = val
        
    return EquationResult(
        equations=eqs,
        parameter_map=pm_dict
    )
