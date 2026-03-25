"""Inventory builder for deterministic audit tooling."""

from __future__ import annotations

import ast
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AtomRecord
from .paths import AGEOA_DIR, AUDIT_MANIFEST_PATH, FIXTURES_DIR
from .upstream import get_repo_revision, get_upstream_mapping

WEAK_TYPES = {"Any", "object", "typing.Any"}
PLACEHOLDER_DOCSTRING_SNIPPETS = {
    "derived deterministically from inputs",
    "skeleton for future ingestion",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _annotation_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _decorator_source(text: str, decorator: ast.AST) -> str:
    return (ast.get_source_segment(text, decorator) or "").strip()


def _has_register_atom(func: ast.FunctionDef, text: str) -> tuple[bool, str | None]:
    for decorator in func.decorator_list:
        source = _decorator_source(text, decorator)
        if "register_atom" in source:
            return True, source
    return False, None


def _function_has_notimplemented(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
            return True
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
    return False


def _top_level_function_details(func: ast.FunctionDef) -> tuple[list[dict[str, Any]], list[str], list[str], bool, bool]:
    details: list[dict[str, Any]] = []
    argument_names: list[str] = []
    required_names: list[str] = []
    positional = list(func.args.posonlyargs) + list(func.args.args)
    defaults = [None] * (len(positional) - len(func.args.defaults)) + list(func.args.defaults)
    for arg, default in zip(positional, defaults):
        annotation = _annotation_text(arg.annotation)
        required = default is None
        details.append(
            {
                "name": arg.arg,
                "required": required,
                "kind": "positional_or_keyword",
                "annotation": annotation,
            }
        )
        argument_names.append(arg.arg)
        if required:
            required_names.append(arg.arg)
    if func.args.vararg is not None:
        details.append(
            {
                "name": func.args.vararg.arg,
                "required": False,
                "kind": "vararg",
                "annotation": _annotation_text(func.args.vararg.annotation),
            }
        )
    for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
        annotation = _annotation_text(arg.annotation)
        required = default is None
        details.append(
            {
                "name": arg.arg,
                "required": required,
                "kind": "keyword_only",
                "annotation": annotation,
            }
        )
        argument_names.append(arg.arg)
        if required:
            required_names.append(arg.arg)
    if func.args.kwarg is not None:
        details.append(
            {
                "name": func.args.kwarg.arg,
                "required": False,
                "kind": "kwargs",
                "annotation": _annotation_text(func.args.kwarg.annotation),
            }
        )
    return details, argument_names, required_names, func.args.vararg is not None, func.args.kwarg is not None


def _build_fixture_index(fixtures_dir: Path = FIXTURES_DIR) -> set[str]:
    atom_keys: set[str] = set()
    if not fixtures_dir.exists():
        return atom_keys
    for fixture_path in fixtures_dir.rglob("*.json"):
        try:
            text = fixture_path.read_text()
        except OSError:
            continue
        marker = '"atom"'
        if marker not in text:
            continue
        for line in text.splitlines():
            if marker not in line:
                continue
            start = line.find(":")
            if start == -1:
                continue
            value = line[start + 1 :].strip().strip('",')
            if ":" in value:
                atom_keys.add(value)
                break
    return atom_keys


def _is_candidate_wrapper_file(path: Path) -> bool:
    if path.name == "__init__.py":
        return False
    if "__pycache__" in path.parts or "ghost" in path.parts:
        return False
    if path.name.startswith("debug_"):
        return False
    return True


def _relative_module_parts(path: Path) -> list[str]:
    rel = path.relative_to(AGEOA_DIR)
    if rel.name == "atoms.py":
        return list(rel.parent.parts)
    return list(rel.with_suffix("").parts)


def _atom_key_for(path: Path, func_name: str) -> str:
    return f"{'/'.join(_relative_module_parts(path))}:{func_name}"


def _atom_name_for(path: Path, func_name: str) -> str:
    return f"ageoa.{'.'.join(_relative_module_parts(path))}.{func_name}"


def _module_import_path_for(path: Path) -> str:
    parts = _relative_module_parts(path)
    if path.name == "atoms.py":
        return "ageoa." + ".".join(parts + ["atoms"])
    return "ageoa." + ".".join(parts)


def _companion_artifacts(path: Path) -> dict[str, bool]:
    if path.name == "atoms.py":
        base = path.parent
        return {
            "has_witnesses": (base / "witnesses.py").exists(),
            "has_state_models": (base / "state_models.py").exists(),
            "has_cdg": (base / "cdg.json").exists(),
            "has_references": (base / "references.json").exists(),
            "has_trace": (base / "trace.jsonl").exists(),
            "has_matches": (base / "matches.json").exists(),
            "has_completed": (base / "COMPLETED.json").exists(),
            "has_failed": (base / "FAILED.json").exists(),
        }
    stem = path.stem
    base = path.parent
    return {
        "has_witnesses": (base / f"{stem}_witnesses.py").exists() or (base / "witnesses.py").exists(),
        "has_state_models": (base / f"{stem}_state.py").exists() or (base / f"{stem}_state_models.py").exists(),
        "has_cdg": (base / f"{stem}_cdg.json").exists() or (base / "cdg.json").exists(),
        "has_references": (base / "references.json").exists(),
        "has_trace": (base / "trace.jsonl").exists(),
        "has_matches": (base / "matches.json").exists(),
        "has_completed": (base / "COMPLETED.json").exists(),
        "has_failed": (base / "FAILED.json").exists(),
    }


def _detect_source_kind(path: Path, skeleton: bool, artifacts: dict[str, bool]) -> str:
    if skeleton:
        return "skeleton"
    generated = artifacts["has_trace"] or artifacts["has_matches"] or artifacts["has_completed"] or artifacts["has_failed"]
    if not generated:
        return "hand_written"
    marker_parts = set(path.parts)
    if any(
        part.endswith("_v2")
        or part.endswith("_d12")
        or "iter" in part
        or part.endswith("_retry")
        or part.endswith("_codex")
        or part.endswith("_sonnet")
        for part in marker_parts
    ):
        return "refined_ingest"
    return "generated_ingest"


def _detect_ffi(path: Path, text: str) -> bool:
    if any(part in {"tempo_jl", "rust_robotics", "bayes_rs"} for part in path.parts):
        return True
    signals = ("ctypes", "cffi", "juliacall", ".dylib", ".so")
    return any(signal in text for signal in signals)


def _detect_stateful(argument_names: list[str], return_annotation: str | None, artifacts: dict[str, bool]) -> bool:
    if artifacts["has_state_models"]:
        return True
    if any(name.endswith("state") or name == "state" for name in argument_names):
        return True
    return bool(return_annotation and "State" in return_annotation)


def _placeholder_witness(binding: str | None) -> bool:
    if not binding:
        return False
    compact = binding.replace(" ", "")
    return "lambda*args,**kwargs:None" in compact or "lambda*args,**kwargs:None)" in compact


def _weak_type_annotations(argument_details: list[dict[str, Any]], return_annotation: str | None) -> list[str]:
    weak: list[str] = []
    for detail in argument_details:
        annotation = detail.get("annotation")
        if annotation in WEAK_TYPES:
            weak.append(f"{detail['name']}:{annotation}")
    if return_annotation in WEAK_TYPES:
        weak.append(f"return:{return_annotation}")
    return weak


def _derive_initial_structural_status(record: AtomRecord) -> str:
    if record.skeleton:
        return "fail"
    if record.require_count == 0 or record.ensure_count == 0:
        return "partial"
    if record.placeholder_witness:
        return "partial"
    return "pass"


def _derive_risk_tier(
    path: Path,
    source_kind: str,
    stateful: bool,
    ffi: bool,
    weak_types: list[str],
    has_parity_tests: bool,
    mapping_found: bool,
    placeholder_witness: bool,
) -> str:
    if source_kind != "hand_written" or stateful or ffi or weak_types or "sklearn" in path.parts or placeholder_witness:
        return "high"
    if not mapping_found or not has_parity_tests or path.name == "atoms.py":
        return "medium"
    return "low"


def discover_atoms() -> tuple[list[AtomRecord], list[dict[str, Any]]]:
    """Discover all registered public atoms under ageoa/."""
    fixture_index = _build_fixture_index()
    atoms: list[AtomRecord] = []
    errors: list[dict[str, Any]] = []

    for path in sorted(AGEOA_DIR.rglob("*.py")):
        if not _is_candidate_wrapper_file(path):
            continue
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            errors.append(
                {
                    "path": str(path.relative_to(AGEOA_DIR.parent)),
                    "error": "syntax_error",
                    "lineno": exc.lineno,
                    "message": exc.msg,
                }
            )
            continue

        artifacts = _companion_artifacts(path)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            has_register, register_source = _has_register_atom(node, text)
            if not has_register:
                continue
            details, argument_names, required_names, uses_varargs, uses_kwargs = _top_level_function_details(node)
            atom_key = _atom_key_for(path, node.name)
            atom_name = _atom_name_for(path, node.name)
            mapping = get_upstream_mapping(atom_key)
            return_annotation = _annotation_text(node.returns)
            weak_types = _weak_type_annotations(details, return_annotation)
            skeleton = _function_has_notimplemented(node)
            source_kind = _detect_source_kind(path, skeleton, artifacts)
            stateful = _detect_stateful(argument_names, return_annotation, artifacts)
            ffi = _detect_ffi(path, text)
            has_docstring = ast.get_docstring(node) is not None
            docstring_summary = None
            if has_docstring:
                docstring_summary = (ast.get_docstring(node) or "").strip().splitlines()[0].strip()
            placeholder_witness = _placeholder_witness(register_source)
            has_witnesses = artifacts["has_witnesses"] or (register_source is not None and not placeholder_witness)
            record = AtomRecord(
                atom_id=f"{atom_name}@{path.relative_to(AGEOA_DIR.parent)}:{node.lineno}",
                atom_name=atom_name,
                atom_key=atom_key,
                module_import_path=_module_import_path_for(path),
                module_path=str(path.relative_to(AGEOA_DIR.parent)),
                wrapper_symbol=node.name,
                wrapper_line=node.lineno,
                domain_family=path.relative_to(AGEOA_DIR).parts[0],
                module_family=_relative_module_parts(path)[0],
                source_kind=source_kind,
                risk_tier="unknown",
                upstream_symbols={} if mapping is None else mapping.to_dict(),
                upstream_version=None,
                source_revision=None if mapping is None else get_repo_revision(mapping.repo),
                review_basis_at=None,
                stateful=stateful,
                ffi=ffi,
                skeleton=skeleton,
                has_state_models=artifacts["has_state_models"],
                has_witnesses=has_witnesses,
                has_cdg=artifacts["has_cdg"],
                has_references=artifacts["has_references"],
                has_parity_tests=atom_key in fixture_index,
                structural_status="unknown",
                runtime_status="unknown",
                semantic_status="unknown",
                developer_semantics_status="unknown",
                parity_test_status="pass" if atom_key in fixture_index else "unknown",
                references_status="pass" if artifacts["has_references"] else "unknown",
                overall_verdict="unknown",
                acceptability_score=None,
                acceptability_band=None,
                max_reviewable_verdict=None,
                argument_names=argument_names,
                required_parameter_names=required_names,
                argument_details=details,
                return_annotation=return_annotation,
                decorator_count=len(node.decorator_list),
                require_count=sum("icontract.require" in _decorator_source(text, dec) for dec in node.decorator_list),
                ensure_count=sum("icontract.ensure" in _decorator_source(text, dec) for dec in node.decorator_list),
                witness_binding=register_source,
                placeholder_witness=placeholder_witness,
                has_docstring=has_docstring,
                docstring_summary=docstring_summary,
                has_weak_types=bool(weak_types),
                weak_type_annotations=weak_types,
                uses_varargs=uses_varargs,
                uses_kwargs=uses_kwargs,
            )
            if docstring_summary and docstring_summary.lower() in PLACEHOLDER_DOCSTRING_SNIPPETS:
                record.inventory_notes.append("placeholder_docstring")
            record.structural_status = _derive_initial_structural_status(record)
            record.risk_tier = _derive_risk_tier(
                path=path,
                source_kind=source_kind,
                stateful=stateful,
                ffi=ffi,
                weak_types=weak_types,
                has_parity_tests=record.has_parity_tests,
                mapping_found=mapping is not None,
                placeholder_witness=placeholder_witness,
            )
            atoms.append(record)
    return atoms, errors


def build_manifest() -> dict[str, Any]:
    """Build the repository-wide audit manifest."""
    atoms, errors = discover_atoms()
    payload = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "repo": "ageo-atoms",
        "atoms": [asdict(record) for record in atoms],
        "inventory_errors": errors,
    }
    return payload


def write_manifest() -> dict[str, Any]:
    """Build and write the repository-wide audit manifest."""
    from .io import write_json

    payload = build_manifest()
    write_json(AUDIT_MANIFEST_PATH, payload)
    return payload
