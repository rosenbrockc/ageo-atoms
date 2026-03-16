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
import json
import logging
import sys
from pathlib import Path

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


def discover_atoms(
    *,
    atom_name: str | None = None,
    domain: str | None = None,
) -> list[tuple[str, Path, object]]:
    """Discover atoms and return (name, directory, callable) triples.

    Returns only atoms whose function accepts and returns np.ndarray.
    """
    results: list[tuple[str, Path, object]] = []

    for atoms_py in sorted(BASE.rglob("atoms.py")):
        d = atoms_py.parent
        if "ghost" in d.parts:
            continue

        if domain and d.parent.name != domain and d.name != domain:
            continue

        # Import the atoms module
        rel = d.relative_to(PROJECT_ROOT)
        module_path = ".".join(rel.parts)
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            logger.debug("Failed to import %s: %s", module_path, exc)
            continue

        # Find @register_atom decorated functions
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            fn = getattr(mod, attr_name, None)
            if not callable(fn):
                continue

            # Check if it has ndarray annotations
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

    Returns a dict suitable for uncertainty.json estimates entry,
    or None if the atom can't be measured.
    """
    if seed is None:
        seed = hash(name) % (2**31)

    base_input = get_base_input(name)

    # Validate the atom works on the base input
    try:
        base_output = np.asarray(fn(base_input))  # type: ignore[operator]
    except Exception as exc:
        logger.warning("Atom %s failed on base input: %s", name, exc)
        return None

    if base_output.ndim == 0:
        logger.info("Atom %s returns scalar — skipping", name)
        return None

    best_estimate: dict | None = None
    best_confidence = -1.0

    for eps in EPSILONS:
        rng = np.random.default_rng(seed)
        successes = 0
        outputs: list[np.ndarray] = []
        inputs_perturbed: list[np.ndarray] = []

        for _ in range(N_TRIALS):
            noise = eps * rng.standard_normal(base_input.shape)
            x_i = base_input + noise
            try:
                y_i = np.asarray(fn(x_i))  # type: ignore[operator]
                if np.any(np.isnan(y_i)) or np.any(np.isinf(y_i)):
                    continue
                outputs.append(y_i.ravel())
                inputs_perturbed.append(x_i.ravel())
                successes += 1
            except Exception:
                continue

        ratio = successes / N_TRIALS
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
        # Confidence scales from trial success ratio
        confidence = float(min(0.9, ratio * 0.9))

        if confidence > best_confidence:
            best_confidence = confidence
            best_estimate = {
                "mode": "empirical",
                "scalar_factor": round(factor, 6),
                "confidence": round(confidence, 3),
                "n_trials": successes,
                "epsilon": eps,
                "input_regime": f"standard_normal({base_input.shape})",
                "notes": f"perturbation harness, {successes}/{N_TRIALS} trials succeeded",
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
        print(f"factor={factor:.4f}  confidence={conf:.2f}  -> {path.relative_to(PROJECT_ROOT)}")
        measured += 1

    print(f"\nDone: {measured} measured, {skipped} skipped")


if __name__ == "__main__":
    main()
