#!/usr/bin/env python3
"""Empirical perturbation harness for measuring atom uncertainty factors.

Usage:
    python scripts/measure_uncertainty.py --atom fft
    python scripts/measure_uncertainty.py --domain numpy
    python scripts/measure_uncertainty.py --all

For each atom with an np.ndarray -> np.ndarray signature, runs a
perturbation analysis and writes uncertainty.json to the atom directory.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import get_args, get_origin

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_base_inputs import get_base_input  # noqa: E402

logger = logging.getLogger(__name__)

BASE = PROJECT_ROOT / "ageoa"

EPSILONS = [1e-8, 1e-6, 1e-4, 1e-2]
N_TRIALS = 500
FAILURE_THRESHOLD = 0.5  # skip if >50% trials fail

# ---------------------------------------------------------------------------
# Default scalar values by parameter name pattern
# ---------------------------------------------------------------------------

_SCALAR_DEFAULTS: dict[str, float | int | str | bool] = {
    # Sampling / frequency
    "sampling_rate": 250.0,
    "fs": 250.0,
    "sample_rate": 250.0,
    "sr": 250.0,
    # Filter
    "order": 4,
    "n": 256,
    "cutoff": 0.1,
    "cutoff_frequency": 0.1,
    "wn": 0.1,
    "low": 0.05,
    "high": 0.4,
    "lowcut": 0.05,
    "highcut": 0.4,
    # Windowing / durations
    "peakwindow": 0.5,
    "beatwindow": 1.0,
    "beatoffset": 0.0,
    "mindelay": 0.4,
    "window_size": 100,
    "size": 100,
    "alarm_size": 50,
    # Thresholds
    "threshold": 0.5,
    "atol": 1e-6,
    "tol": 1e-6,
    "transition_threshold": 0.5,
    "rtol": 1e-5,
    # Regularization
    "smoothing": 0.1,
    "epsilon": 1.0,
    "alpha": 0.01,
    "beta": 1.0,
    "gamma": 1.0,
    "lambda_": 0.01,
    # Optimization
    "maxiter": 100,
    "popsize": 15,
    "mutation": 0.5,
    "recombination": 0.7,
    # Steps
    "step_size": 0.01,
    "dt": 0.01,
    "time_step": 0.01,
    # Misc scalars
    "degree": 3,
    "k": 3,
    "neighbors": 10,
    "num_components": 2,
    "n_components": 2,
    "n_iter": 100,
    "iterations": 100,
    "max_iterations": 100,
    "seed": 42,
    # String / choice params
    "mode": "time",
    "norm": "ortho",
    "bc_type": "not-a-knot",
    "kernel": "linear",
    "strategy": "best1bin",
    "updating": "immediate",
    "side": "left",
    "btype": "low",
    "ftype": "butter",
    # Boolean
    "disp": False,
    "polish": False,
    "verbose": False,
    "normalize": False,
    "correct": True,
    "detrend": True,
}

# Type-based fallbacks when name isn't in the dict
_TYPE_DEFAULTS = {
    "float": 1.0,
    "int": 4,
    "str": "default",
    "bool": False,
}


def _generate_default_for_param(
    name: str,
    annotation: type | str | None,
    rng: np.random.Generator,
) -> object:
    """Generate a reasonable default value for a function parameter.

    Priority: name-based lookup > type-based inference > generic ndarray.
    """
    # Check name-based defaults first
    if name in _SCALAR_DEFAULTS:
        return _SCALAR_DEFAULTS[name]

    # Parse annotation string
    ann_str = str(annotation).lower() if annotation else ""

    # ndarray / array types -> generate array
    if any(tok in ann_str for tok in ("ndarray", "ndarr", "array")):
        # Check if it's likely a matrix param
        matrix_names = {"F", "H", "Q", "R", "P", "B", "A", "matrix", "covariance"}
        if name in matrix_names or "matrix" in name.lower() or "covariance" in name.lower():
            m = rng.standard_normal((4, 4))
            return m @ m.T + np.eye(4)  # positive definite
        return rng.standard_normal(256)

    # Callable -> identity
    if "callable" in ann_str:
        return lambda x: x

    # dict -> empty dict
    if "dict" in ann_str:
        return {}

    # list of tuples (bounds)
    if "list" in ann_str and "tuple" in ann_str:
        return [(-5.0, 5.0)] * 4

    # tuple -> generic tuple
    if "tuple" in ann_str:
        return (0.0, 1.0)

    # NoneType / Optional -> None
    if annotation is None or "none" in ann_str:
        return None

    # float / int / str / bool
    for type_name, default in _TYPE_DEFAULTS.items():
        if type_name in ann_str:
            return default

    # Last resort: check name heuristics
    lowered = name.lower()
    if any(kw in lowered for kw in ("rate", "freq", "hz")):
        return 250.0
    if any(kw in lowered for kw in ("order", "degree", "size", "count", "num")):
        return 4
    if any(kw in lowered for kw in ("threshold", "tol", "eps")):
        return 0.5

    # Truly unknown -> try None, then 1.0
    return None


def _build_call_args(
    fn: object,
    rng: np.random.Generator,
    atom_name: str,
) -> tuple[dict[str, object], list[str]] | None:
    """Introspect fn's signature and build a dict of keyword arguments.

    Returns (kwargs_dict, ndarray_param_names) or None if introspection fails.
    """
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None

    kwargs: dict[str, object] = {}
    ndarray_params: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue

        ann = param.annotation
        if ann is inspect.Parameter.empty:
            ann = None

        # If there's a default, use it
        if param.default is not inspect.Parameter.empty:
            kwargs[pname] = param.default
            # Still track if it's an ndarray param
            ann_str = str(ann).lower() if ann else ""
            if any(tok in ann_str for tok in ("ndarray", "ndarr", "array")):
                # Override None defaults with actual arrays
                if kwargs[pname] is None:
                    kwargs[pname] = rng.standard_normal(256)
                ndarray_params.append(pname)
            continue

        # No default — generate one
        val = _generate_default_for_param(pname, ann, rng)
        kwargs[pname] = val

        ann_str = str(ann).lower() if ann else ""
        if isinstance(val, np.ndarray) or any(tok in ann_str for tok in ("ndarray", "ndarr", "array")):
            if not isinstance(val, np.ndarray):
                kwargs[pname] = rng.standard_normal(256)
            ndarray_params.append(pname)

    # If first param has no annotation and no ndarray params found, assume it's the primary signal
    if not ndarray_params:
        first = next(iter(sig.parameters.values()), None)
        if first is not None and first.name not in ("self", "cls"):
            kwargs[first.name] = get_base_input(atom_name)
            ndarray_params.append(first.name)

    return kwargs, ndarray_params


def discover_atoms(
    *,
    atom_name: str | None = None,
    domain: str | None = None,
) -> list[tuple[str, Path, object]]:
    """Discover atoms and return (name, directory, callable) triples.

    Returns only atoms whose function accepts or returns np.ndarray.
    """
    results: list[tuple[str, Path, object]] = []

    for atoms_py in sorted(BASE.rglob("atoms.py")):
        d = atoms_py.parent
        if "ghost" in d.parts:
            continue

        if domain and d.parent.name != domain and d.name != domain:
            # Also check grandparent for nested domains like numpy/fft_v2
            if not any(p.name == domain for p in d.parents):
                continue

        # Import the atoms module
        rel = d.relative_to(PROJECT_ROOT)
        module_path = ".".join(rel.parts)
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            logger.debug("Failed to import %s: %s", module_path, exc)
            continue

        # Find callable atoms with ndarray in annotations
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            fn = getattr(mod, attr_name, None)
            if not callable(fn):
                continue

            annotations = getattr(fn, "__annotations__", {})
            if not annotations:
                continue

            has_ndarray = any(
                "ndarray" in str(ann) for ann in annotations.values()
            )
            if not has_ndarray:
                continue

            if atom_name and attr_name != atom_name:
                continue

            results.append((attr_name, d, fn))

    return results


def measure_atom(
    name: str,
    fn: object,
    *,
    seed: int | None = None,
) -> dict | None:
    """Run perturbation analysis for a single atom.

    Introspects function signature to generate appropriate multi-arg inputs.
    Only ndarray parameters are perturbed; scalars are held constant.

    Returns a dict suitable for uncertainty.json estimates entry,
    or None if the atom can't be measured.
    """
    if seed is None:
        seed = hash(name) % (2**31)

    rng = np.random.default_rng(seed)

    # Build base call args from signature introspection
    result = _build_call_args(fn, rng, name)
    if result is None:
        logger.warning("Atom %s: could not introspect signature", name)
        return None

    base_kwargs, ndarray_params = result

    if not ndarray_params:
        logger.info("Atom %s: no ndarray parameters found", name)
        return None

    # Override known atoms' primary array from the registry
    primary = ndarray_params[0]
    registered_input = get_base_input(name)
    # Only use registry input if it matches expected dimensionality
    if primary in base_kwargs:
        existing = base_kwargs[primary]
        if isinstance(existing, np.ndarray) and existing.ndim == registered_input.ndim:
            base_kwargs[primary] = registered_input

    # Validate the atom works with base inputs
    try:
        base_output = np.asarray(fn(**base_kwargs))  # type: ignore[operator]
    except Exception as exc:
        logger.warning("Atom %s failed on base input: %s", name, exc)
        return None

    if base_output.ndim == 0:
        logger.info("Atom %s returns scalar — skipping", name)
        return None

    best_estimate: dict | None = None
    best_confidence = -1.0

    for eps in EPSILONS:
        trial_rng = np.random.default_rng(seed)
        successes = 0
        outputs: list[np.ndarray] = []
        inputs_perturbed: list[np.ndarray] = []

        for _ in range(N_TRIALS):
            # Perturb all ndarray params
            perturbed_kwargs = dict(base_kwargs)
            perturbed_arrays: list[np.ndarray] = []
            for pname in ndarray_params:
                base_arr = base_kwargs[pname]
                if not isinstance(base_arr, np.ndarray):
                    continue
                noise = eps * trial_rng.standard_normal(base_arr.shape)
                p_arr = base_arr + noise
                perturbed_kwargs[pname] = p_arr
                perturbed_arrays.append(p_arr.ravel())

            if not perturbed_arrays:
                break

            try:
                y_i = np.asarray(fn(**perturbed_kwargs))  # type: ignore[operator]
                if np.any(np.isnan(y_i)) or np.any(np.isinf(y_i)):
                    continue
                outputs.append(y_i.ravel())
                inputs_perturbed.append(np.concatenate(perturbed_arrays))
                successes += 1
            except Exception:
                continue

        ratio = successes / N_TRIALS if N_TRIALS > 0 else 0
        if ratio < FAILURE_THRESHOLD:
            logger.info(
                "Atom %s: eps=%e skipped (%.0f%% failures)",
                name, eps, (1 - ratio) * 100,
            )
            continue

        # Compute factor = mean(std(Y)) / mean(std(X_perturbed))
        Y = np.array(outputs)
        X = np.array(inputs_perturbed)

        y_std = np.mean(np.std(Y, axis=0))
        x_std = np.mean(np.std(X, axis=0))

        if x_std < 1e-15:
            continue

        factor = float(y_std / x_std)
        confidence = float(min(0.9, ratio * 0.9))

        if confidence > best_confidence:
            best_confidence = confidence
            shapes = {p: str(base_kwargs[p].shape) for p in ndarray_params  # type: ignore[union-attr]
                      if isinstance(base_kwargs[p], np.ndarray)}
            best_estimate = {
                "mode": "empirical",
                "scalar_factor": round(factor, 6),
                "confidence": round(confidence, 3),
                "n_trials": successes,
                "epsilon": eps,
                "input_regime": f"multi-arg perturbation, shapes={shapes}",
                "notes": f"perturbation harness, {successes}/{N_TRIALS} trials, "
                         f"perturbed params: {ndarray_params}",
            }

    return best_estimate


def write_uncertainty_json(
    atom_name: str,
    atom_dir: Path,
    estimate: dict,
) -> Path:
    """Write uncertainty.json to the atom directory."""
    uj_path = atom_dir / "uncertainty.json"

    # If file exists, merge estimates (keep existing, add new)
    data: dict
    if uj_path.exists():
        try:
            data = json.loads(uj_path.read_text())
        except json.JSONDecodeError:
            data = {"atom": atom_name, "estimates": []}
    else:
        data = {"atom": atom_name, "estimates": []}

    # Replace any existing empirical estimate, keep others
    data["estimates"] = [
        e for e in data.get("estimates", []) if e.get("mode") != "empirical"
    ]
    data["estimates"].append(estimate)

    uj_path.write_text(json.dumps(data, indent=2) + "\n")
    return uj_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure atom uncertainty factors")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--atom", help="Measure a specific atom by name")
    group.add_argument("--domain", help="Measure all atoms in a domain")
    group.add_argument("--all", action="store_true", help="Measure all ndarray atoms")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    atoms = discover_atoms(
        atom_name=args.atom,
        domain=args.domain,
    )

    if not atoms:
        print("No matching atoms found.")
        sys.exit(1)

    print(f"Measuring {len(atoms)} atom(s)...\n")

    measured = 0
    skipped = 0
    outliers: list[tuple[str, float]] = []
    for name, atom_dir, fn in atoms:
        print(f"  {name}...", end=" ", flush=True)
        estimate = measure_atom(name, fn)
        if estimate is None:
            print("SKIP")
            skipped += 1
            continue

        path = write_uncertainty_json(name, atom_dir, estimate)
        factor = estimate["scalar_factor"]
        conf = estimate["confidence"]
        tag = ""
        if factor > 10.0:
            tag = " [OUTLIER: factor > 10]"
            outliers.append((name, factor))
        elif factor < 0.1:
            tag = " [OUTLIER: factor < 0.1]"
            outliers.append((name, factor))
        print(f"factor={factor:.4f}  confidence={conf:.2f}  -> {path.relative_to(PROJECT_ROOT)}{tag}")
        measured += 1

    print(f"\nDone: {measured} measured, {skipped} skipped")
    if outliers:
        print(f"\nOutliers ({len(outliers)}):")
        for oname, ofactor in outliers:
            print(f"  {oname}: factor={ofactor:.4f}")


if __name__ == "__main__":
    main()
