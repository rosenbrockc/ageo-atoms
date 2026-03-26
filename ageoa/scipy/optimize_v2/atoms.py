from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


from typing import Callable

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from scipy.optimize import OptimizeResult
from .witnesses import witness_shgoglobaloptimization, witness_differentialevolutionoptimization


@register_atom(witness_shgoglobaloptimization)
@icontract.require(lambda func: func is not None, "func cannot be None")
@icontract.require(lambda bounds: bounds is not None, "bounds cannot be None")
@icontract.require(lambda args: args is not None, "args cannot be None")
@icontract.require(lambda constraints: constraints is not None, "constraints cannot be None")
@icontract.require(lambda n: n is not None, "n cannot be None")
@icontract.require(lambda iters: iters is not None, "iters cannot be None")
@icontract.require(lambda minimizer_kwargs: minimizer_kwargs is not None, "minimizer_kwargs cannot be None")
@icontract.require(lambda options: options is not None, "options cannot be None")
@icontract.require(lambda sampling_method: sampling_method is not None, "sampling_method cannot be None")
@icontract.ensure(lambda result: result is not None, "ShgoGlobalOptimization output must not be None")
def shgoglobaloptimization(func: Callable[..., float], bounds: list[tuple[float, float]], args: tuple, constraints: list[dict] | dict, n: int, iters: int, callback: Callable | None, minimizer_kwargs: dict, options: dict, sampling_method: str | Callable) -> OptimizeResult:
    """Finds the global minimum of a scalar function using Simplicial Homology Global Optimization (SHGO): iteratively samples the bounded search space via simplicial or quasi-random methods, builds a simplicial complex to locate local minima candidates, and refines each candidate with a local minimizer.

    Args:
        func: must return a scalar float
        bounds: length equals problem dimensionality
        args: extra arguments passed to func
        constraints: scipy-style constraint dicts or NonlinearConstraint
        n: n >= 1
        iters: iters >= 1
        callback: called after each iteration; None to skip
        minimizer_kwargs: keyword arguments passed to the local minimizer
        options: see config_params
        sampling_method: one of 'simplicial', 'halton', 'sobol', or a custom callable

    Returns:
        OptimizeResult with x, fun, and convergence metadata.
    """
    from scipy.optimize import shgo as _shgo
    return _shgo(func, bounds, args=args, constraints=constraints, n=n, iters=iters, callback=callback, minimizer_kwargs=minimizer_kwargs, options=options, sampling_method=sampling_method)


@register_atom(witness_differentialevolutionoptimization)
@icontract.require(lambda mutation: isinstance(mutation, (float, int, np.number, tuple)), "mutation must be numeric or tuple")
@icontract.require(lambda recombination: isinstance(recombination, (float, int, np.number)), "recombination must be numeric")
@icontract.require(lambda atol: isinstance(atol, (float, int, np.number)), "atol must be numeric")
@icontract.ensure(lambda result: result is not None, "DifferentialEvolutionOptimization output must not be None")
def differentialevolutionoptimization(func: Callable[..., float], bounds: list[tuple[float, float]], args: tuple, strategy: str, maxiter: int, popsize: int, tol: float, mutation: float | tuple[float, float], recombination: float, seed: int | np.random.RandomState | None, callback: Callable | None, disp: bool, polish: bool, init: str | np.ndarray, atol: float, updating: str, workers: int | Callable, constraints: list[dict] | dict | None, x0: np.ndarray | None) -> OptimizeResult:
    """Finds the global minimum of a scalar function using Differential Evolution (DE), a population-based optimization method. Applies stochastic mutation and crossover each generation to explore the bounded search space.

    Args:
        func: must return a scalar float
        bounds: length equals problem dimensionality
        args: extra arguments passed to func
        strategy: one of 'best1bin', 'best1exp', 'rand1exp', 'randtobest1bin', 'currenttobest1bin', 'best2exp', 'rand2exp', 'randtobest1exp', 'currenttobest1exp', 'best2bin', 'rand2bin', 'rand1bin'
        maxiter: maxiter >= 1
        popsize: popsize >= 1
        tol: tol >= 0
        mutation: scalar in [0, 2] or tuple (min_F, max_F)
        recombination: in [0, 1]
        seed: random state or seed integer; None for default
        callback: return True to halt early; None to skip
        disp: whether to print convergence messages
        polish: whether to polish the best result with L-BFGS-B
        init: 'latinhypercube', 'sobol', 'halton', 'random', or ndarray of shape (popsize*N, N)
        atol: atol >= 0
        updating: 'immediate' or 'deferred'
        workers: -1 uses all CPU cores; map-like must conform to Pool.map interface
        constraints: scipy-style constraint dicts or None
        x0: shape (N,); initial guess or None

    Returns:
        OptimizeResult with x, fun, and convergence metadata.
    """
    from scipy.optimize import differential_evolution as _de
    return _de(func, bounds, args=args, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation, recombination=recombination, seed=seed, callback=callback, disp=disp, polish=polish, init=init, atol=atol, updating=updating, workers=workers, constraints=constraints, x0=x0)
