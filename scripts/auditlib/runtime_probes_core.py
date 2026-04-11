"""Shared core for conservative deterministic runtime probes."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ageoa_julia_runtime import configure_juliacall_env

from .io import safe_atom_stem, write_json
from .paths import AUDIT_PROBES_DIR, ROOT
from .semantics import utc_now, write_evidence_section

configure_juliacall_env()


@dataclass(frozen=True)
class ProbeCase:
    """A deterministic positive or negative runtime probe."""

    description: str
    invoke: Callable[[Callable[..., Any]], Any]
    validate: Callable[[Any], None] | None = None
    expect_exception: bool = False


@dataclass(frozen=True)
class ProbePlan:
    """Probe plan for one allowlisted atom."""

    positive: ProbeCase
    negative: ProbeCase | None = None
    parity_used: bool = False


def install_ageoa_stub(root: Path = ROOT) -> None:
    """Install a lightweight `ageoa` package stub so probes avoid ageoa.__init__."""
    ageoa_dir = root / "ageoa"
    existing = sys.modules.get("ageoa")
    if existing is not None and getattr(existing, "__path__", None):
        return
    stub = types.ModuleType("ageoa")
    stub.__path__ = [str(ageoa_dir)]
    stub.__package__ = "ageoa"
    sys.modules["ageoa"] = stub


def install_package_stub(package_name: str, root: Path = ROOT) -> None:
    """Install a lightweight package stub for an ageoa subpackage."""
    if package_name == "ageoa":
        install_ageoa_stub(root)
        return
    if not package_name.startswith("ageoa."):
        return
    parent_name = package_name.rsplit(".", 1)[0]
    install_package_stub(parent_name, root)
    existing = sys.modules.get(package_name)
    if existing is not None and getattr(existing, "__path__", None):
        return
    package_dir = root / Path(*package_name.split("."))
    stub = types.ModuleType(package_name)
    stub.__path__ = [str(package_dir)]
    stub.__package__ = package_name
    sys.modules[package_name] = stub


def _load_alias_module(alias_name: str, alias_file: Path) -> Any:
    spec = importlib.util.spec_from_file_location(alias_name, alias_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for alias module {alias_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = module
    spec.loader.exec_module(module)
    return module


def _install_legacy_sibling_aliases(module_file: Path) -> list[str]:
    installed: list[str] = []
    legacy_candidates = {
        "state_models": module_file.with_name(f"{module_file.stem}_state.py"),
        "witnesses": module_file.with_name(f"{module_file.stem}_witnesses.py"),
    }
    for alias_name, alias_file in legacy_candidates.items():
        if alias_name in sys.modules or not alias_file.exists():
            continue
        _load_alias_module(alias_name, alias_file)
        installed.append(alias_name)
    return installed


def load_module_from_file(module_import_path: str, module_file: Path) -> Any:
    """Load a module directly from its source file while preserving package-relative imports."""
    package_name = module_import_path.rsplit(".", 1)[0]
    install_package_stub(package_name)
    spec = importlib.util.spec_from_file_location(module_import_path, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_import_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_import_path] = module
    module_dir = str(module_file.parent)
    added_sys_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_sys_path = True
    alias_names = _install_legacy_sibling_aliases(module_file)
    try:
        spec.loader.exec_module(module)
    finally:
        for alias_name in alias_names:
            sys.modules.pop(alias_name, None)
        if added_sys_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
    return module


def safe_import_module(module_import_path: str) -> Any:
    """Import an ageoa submodule without executing ageoa.__init__."""
    install_ageoa_stub()
    try:
        return importlib.import_module(module_import_path)
    except Exception:
        if not module_import_path.startswith("ageoa."):
            raise
        module_file = ROOT / Path(*module_import_path.split("."))
        module_file = module_file.with_suffix(".py")
        if not module_file.exists():
            raise
        return load_module_from_file(module_import_path, module_file)


def _summarize_value(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "length": len(value)}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__, "repr": repr(value)[:120]}


def _run_case(func: Callable[..., Any], case: ProbeCase | None) -> dict[str, Any]:
    if case is None:
        return {"status": "not_applicable", "description": None}
    try:
        result = case.invoke(func)
        if case.expect_exception:
            return {
                "status": "fail",
                "description": case.description,
                "message": "probe unexpectedly succeeded",
                "result_summary": _summarize_value(result),
            }
        if case.validate is not None:
            case.validate(result)
        return {
            "status": "pass",
            "description": case.description,
            "result_summary": _summarize_value(result),
        }
    except Exception as exc:
        if case.expect_exception:
            return {
                "status": "pass",
                "description": case.description,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:240],
            }
        return {
            "status": "fail",
            "description": case.description,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:240],
        }


_PROBE_PLANS: dict[str, ProbePlan] = {}


def assemble_probe_plans(*plan_groups: dict[str, ProbePlan]) -> dict[str, ProbePlan]:
    """Merge one or more probe plan dictionaries, rejecting duplicate keys."""
    merged: dict[str, ProbePlan] = {}
    for group in plan_groups:
        for atom_name, plan in group.items():
            if atom_name in merged:
                raise ValueError(f"Duplicate runtime probe plan for {atom_name}")
            merged[atom_name] = plan
    return merged


def set_probe_plans(probe_plans: dict[str, ProbePlan]) -> None:
    """Install the active probe plan registry used by build_runtime_probe()."""
    global _PROBE_PLANS
    _PROBE_PLANS = dict(probe_plans)


def get_probe_plans() -> dict[str, ProbePlan]:
    return dict(_PROBE_PLANS)


def build_runtime_probe(record: dict[str, Any]) -> dict[str, Any]:
    """Run the safe deterministic probe plan for one atom, or skip it."""
    base = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "atom_id": record["atom_id"],
        "atom_name": record["atom_name"],
        "probe_status": "skipped",
        "positive_probe": {"status": "not_applicable"},
        "negative_probe": {"status": "not_applicable"},
        "parity_used": False,
        "skip_reason": None,
        "exception_type": None,
        "exception_message": None,
    }
    if record.get("skeleton"):
        return {
            "status": "fail",
            "findings": ["RUNTIME_NOT_IMPLEMENTED"],
            "notes": ["Wrapper is a skeleton or raises NotImplementedError."],
            "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
            **base,
            "skip_reason": "skeleton_wrapper",
        }

    plan = _PROBE_PLANS.get(record["atom_name"])
    if plan is None:
        return {
            "status": "not_applicable",
            "findings": ["RUNTIME_PROBE_SKIPPED"],
            "notes": ["Atom is outside the conservative safe probe allowlist."],
            "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
            **base,
            "skip_reason": "unsupported_scope",
        }

    try:
        module = safe_import_module(record["module_import_path"])
        func = getattr(module, record["wrapper_symbol"])
    except Exception as exc:
        return {
            "status": "partial",
            "findings": ["RUNTIME_IMPORT_FAIL"],
            "notes": ["Import failed before the runtime probe could execute."],
            "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
            **base,
            "probe_status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:240],
        }

    positive = _run_case(func, plan.positive)
    negative = _run_case(func, plan.negative)
    findings: list[str] = []
    notes: list[str] = []

    if positive["status"] == "pass":
        findings.append("RUNTIME_PROBE_PASS")
    else:
        findings.append("RUNTIME_PROBE_FAIL")
        if positive.get("exception_type"):
            notes.append(
                f"Positive probe raised {positive['exception_type']}: {positive.get('exception_message', '')}"
            )

    if negative["status"] == "pass":
        findings.append("RUNTIME_CONTRACT_NEGATIVE_PASS")
    elif negative["status"] == "fail":
        findings.append("RUNTIME_CONTRACT_NEGATIVE_FAIL")
        if negative.get("exception_type"):
            notes.append(
                f"Negative probe raised {negative['exception_type']}: {negative.get('exception_message', '')}"
            )

    if positive["status"] == "fail":
        status = "fail"
    elif negative["status"] == "fail":
        status = "fail"
    elif negative["status"] == "not_applicable":
        status = "partial"
    else:
        status = "pass"

    return {
        "status": status,
        "findings": findings,
        "notes": notes,
        "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
        **base,
        "probe_status": "executed",
        "positive_probe": positive,
        "negative_probe": negative,
        "parity_used": plan.parity_used,
        "exception_type": positive.get("exception_type"),
        "exception_message": positive.get("exception_message"),
    }


def write_runtime_probe(record: dict[str, Any]) -> dict[str, Any]:
    """Run, persist, and merge runtime probe evidence for one atom."""
    section = build_runtime_probe(record)
    write_json(AUDIT_PROBES_DIR / f"{safe_atom_stem(record['atom_id'])}.json", section)
    write_evidence_section(record["atom_id"], "runtime_probe", section)
    return section
