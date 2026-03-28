"""Upstream mapping and signature extraction helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .io import ensure_dir
from .models import UpstreamMapping
from .paths import ATOM_MANIFEST_PATH, AUDIT_DIR, THIRD_PARTY_DIR

_MPLCONFIGDIR = AUDIT_DIR / "mplconfig"
ensure_dir(_MPLCONFIGDIR)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


@lru_cache(maxsize=1)
def load_atom_manifest() -> dict[str, UpstreamMapping]:
    """Load the repo-local atom manifest."""
    raw = yaml.safe_load(ATOM_MANIFEST_PATH.read_text()) or []
    mapping: dict[str, UpstreamMapping] = {}
    for item in raw:
        atom_key = str(item.get("atom", "")).strip()
        upstream = item.get("upstream") or {}
        mapping[atom_key] = UpstreamMapping(
            repo=upstream.get("repo"),
            module=upstream.get("module"),
            function=upstream.get("function"),
            language=upstream.get("language"),
            notes=upstream.get("notes") or item.get("notes"),
        )
    return mapping


def get_upstream_mapping(atom_key: str) -> UpstreamMapping | None:
    """Look up an atom key in the deterministic atom manifest."""
    return load_atom_manifest().get(atom_key)


@lru_cache(maxsize=None)
def get_repo_revision(repo_name: str | None) -> str | None:
    """Read a vendored repo git revision if present."""
    if not repo_name or repo_name == "~":
        return None
    repo_path = THIRD_PARTY_DIR / repo_name
    if not repo_path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


@lru_cache(maxsize=None)
def get_installed_package_version(module_name: str | None) -> str | None:
    """Resolve an installed package version for import-based upstream mappings."""
    if not module_name:
        return None
    root = module_name.split(".", 1)[0]
    try:
        return importlib.metadata.version(root)
    except Exception:
        return None


def _module_to_candidate_paths(repo_name: str, module_name: str) -> list[Path]:
    parts = module_name.split(".")
    repo_root = THIRD_PARTY_DIR / repo_name
    candidates = [
        repo_root / Path(*parts).with_suffix(".py"),
        repo_root / Path(*parts) / "__init__.py",
    ]
    if len(parts) > 1:
        candidates.append(repo_root / Path(*parts[1:]).with_suffix(".py"))
        candidates.append(repo_root / Path(*parts[1:]) / "__init__.py")
    if repo_name.endswith(".jl") and module_name:
        candidates.append(repo_root / "src" / f"{parts[-1]}.jl")
    return candidates


def resolve_vendored_module_path(mapping: UpstreamMapping) -> Path | None:
    """Resolve a vendored source path for a mapped upstream symbol."""
    if not mapping.repo or not mapping.module or not mapping.language:
        return None
    if mapping.language != "python":
        return None
    for candidate in _module_to_candidate_paths(mapping.repo, mapping.module):
        if candidate.exists():
            return candidate
    return None


def _annotation_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_signature_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        args.append(
            {
                "name": arg.arg,
                "required": default is None,
                "kind": "positional_or_keyword",
                "annotation": _annotation_text(arg.annotation),
            }
        )
    if node.args.vararg is not None:
        args.append(
            {
                "name": node.args.vararg.arg,
                "required": False,
                "kind": "vararg",
                "annotation": _annotation_text(node.args.vararg.annotation),
            }
        )
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        args.append(
            {
                "name": arg.arg,
                "required": default is None,
                "kind": "keyword_only",
                "annotation": _annotation_text(arg.annotation),
            }
        )
    if node.args.kwarg is not None:
        args.append(
            {
                "name": node.args.kwarg.arg,
                "required": False,
                "kind": "kwargs",
                "annotation": _annotation_text(node.args.kwarg.annotation),
            }
        )
    return {
        "parameter_names": [arg["name"] for arg in args if arg["kind"] not in {"vararg", "kwargs"}],
        "required_parameter_names": [arg["name"] for arg in args if arg["required"]],
        "parameters": args,
        "return_annotation": _annotation_text(node.returns),
    }


def _extract_from_ast(path: Path, qualname: str) -> dict[str, Any] | None:
    text = path.read_text()
    tree = ast.parse(text)
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                return _extract_signature_from_function(node)
        return None
    if len(parts) == 2:
        class_name, method_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                        return _extract_signature_from_function(child)
    return None


def _extract_with_inspect(mapping: UpstreamMapping) -> dict[str, Any] | None:
    if not mapping.module or not mapping.function:
        return None
    try:
        module = importlib.import_module(mapping.module)
    except Exception:
        return None
    target: Any = module
    try:
        for part in mapping.function.split("."):
            target = getattr(target, part)
    except Exception:
        return None
    try:
        sig = inspect.signature(target)
    except Exception:
        return None
    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        kind_name = param.kind.name.lower()
        if kind_name == "positional_or_keyword":
            kind = "positional_or_keyword"
        elif kind_name == "var_positional":
            kind = "var_positional"
        elif kind_name == "var_keyword":
            kind = "var_keyword"
        elif kind_name == "keyword_only":
            kind = "keyword_only"
        elif kind_name == "positional_only":
            kind = "positional_only"
        else:
            kind = kind_name
        required = param.default is inspect._empty and kind not in {"var_positional", "var_keyword"}
        params.append(
            {
                "name": name,
                "required": required,
                "kind": kind,
                "annotation": None if param.annotation is inspect._empty else repr(param.annotation),
            }
        )
    return {
        "parameter_names": [param["name"] for param in params if param["kind"] not in {"var_positional", "var_keyword"}],
        "required_parameter_names": [param["name"] for param in params if param["required"]],
        "parameters": params,
        "return_annotation": None if sig.return_annotation is inspect._empty else repr(sig.return_annotation),
    }


def resolve_upstream_signature(mapping: UpstreamMapping) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Resolve a deterministic upstream signature."""
    if not mapping.module or not mapping.function or not mapping.language:
        return None, None, None
    if mapping.language != "python":
        return None, None, "non_python_upstream"
    vendored_path = resolve_vendored_module_path(mapping)
    if vendored_path is not None:
        signature = _extract_from_ast(vendored_path, mapping.function)
        if signature is not None:
            return signature, "vendored_ast", str(vendored_path)
    inspected = _extract_with_inspect(mapping)
    if inspected is not None:
        return inspected, "inspect", mapping.module
    return None, None, "signature_unavailable"
