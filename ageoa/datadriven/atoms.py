"""DataDrivenDiffEq SINDy Macro-Atoms."""

from __future__ import annotations

import re
from typing import Any, List

import icontract
import numpy as np
from pydantic import BaseModel, Field

from ageoa.datadriven.witnesses import witness_discover_equations
from ageoa.ghost.registry import register_atom

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_jl: Any | None = None
_datadriven_loaded = False


def _get_jl() -> Any:
    """Import juliacall lazily and load DataDriven packages once."""
    global _jl, _datadriven_loaded
    if _jl is None:
        from juliacall import Main as jl_main

        _jl = jl_main
    if not _datadriven_loaded:
        _jl.seval("using DataDrivenDiffEq")
        _jl.seval("using DataDrivenSparse")
        _jl.seval("using ModelingToolkit")
        _datadriven_loaded = True
    return _jl


def _validate_variable_names(variable_names: List[str]) -> List[str]:
    if not variable_names:
        raise ValueError("variable_names must be non-empty")

    names: List[str] = []
    for raw in variable_names:
        name = str(raw).strip()
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError(
                f"Invalid Julia identifier '{raw}'. "
                "Use [A-Za-z_][A-Za-z0-9_]* names only."
            )
        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("variable_names must be unique")

    return names


class EquationResult(BaseModel):
    """Result container for symbolic equation discovery."""

    equations: List[str] = Field(
        default_factory=list, description="The discovered symbolic equations."
    )
    parameter_map: dict[str, float] = Field(
        default_factory=dict, description="Map of parameter values discovered."
    )


@register_atom(witness_discover_equations)
@icontract.require(lambda X: X.ndim == 2, "X must be a 2D array (features x samples)")
@icontract.require(lambda Y: Y.ndim == 2, "Y must be a 2D array (targets x samples)")
@icontract.require(
    lambda X, Y: X.shape[1] == Y.shape[1], "X and Y must have same number of samples"
)
@icontract.require(
    lambda variable_names: len(variable_names) > 0,
    "variable_names must be non-empty",
)
@icontract.require(lambda max_degree: max_degree > 0, "max_degree must be positive")
@icontract.require(lambda lambda_val: lambda_val > 0, "Sparsity penalty lambda_val must be positive")
@icontract.ensure(lambda result: isinstance(result, EquationResult))
def discover_equations(
    X: np.ndarray,
    Y: np.ndarray,
    variable_names: List[str],
    max_degree: int = 4,
    lambda_val: float = 0.1,
) -> EquationResult:
    """Sparsity-promoting symbolic regression for discovering governing equations from data."""
    names = _validate_variable_names(variable_names)
    if X.shape[0] != len(names):
        raise ValueError(
            f"X feature dimension {X.shape[0]} does not match number "
            f"of variable names {len(names)}"
        )

    jl = _get_jl()
    degree = int(max_degree)

    # Variable interpolation is safe due strict identifier validation above.
    jl.seval(f"ModelingToolkit.@variables {' '.join(names)}")
    jl.seval(f"u_vars = [{', '.join(names)}]")
    jl.seval(f"b = DataDrivenDiffEq.polynomial_basis(u_vars, {degree})")
    basis = jl.seval("DataDrivenDiffEq.Basis(b, u_vars)")

    jl.X_train = X.astype(np.float64)
    jl.Y_train = Y.astype(np.float64)
    prob = jl.seval("DataDrivenDiffEq.DirectDataDrivenProblem(X_train, Y_train)")

    jl.lambda_val = float(lambda_val)
    opt = jl.seval("DataDrivenSparse.ADMM(lambda_val)")

    jl.prob = prob
    jl.basis = basis
    jl.opt = opt
    res = jl.seval("DataDrivenDiffEq.solve(prob, basis, opt)")
    jl.res = res

    basis_res = jl.seval("DataDrivenDiffEq.get_basis(res)")
    jl.basis_res = basis_res
    equations = [line.strip() for line in str(basis_res).splitlines() if line.strip()]

    param_map = jl.seval("DataDrivenDiffEq.get_parameter_map(basis_res)")
    params: dict[str, float] = {}
    for pair in param_map:
        params[str(pair[0])] = float(pair[1])

    return EquationResult(equations=equations, parameter_map=params)
