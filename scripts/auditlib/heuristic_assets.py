"""Heuristic asset policy helpers for cross-family audit validation."""

from __future__ import annotations

import re
from typing import Any

HEURISTIC_SOURCE_KINDS = {"generated_ingest", "refined_ingest", "skeleton"}
PLACEHOLDER_DOC_PATTERNS = (
    "auto-generated",
    "derived deterministically from inputs",
    "skeleton for future ingestion",
    "todo",
    "tbd",
)
ACTION_VERBS = {
    "apply",
    "build",
    "calculate",
    "compute",
    "convert",
    "correct",
    "detect",
    "estimate",
    "extract",
    "filter",
    "fit",
    "generate",
    "measure",
    "normalize",
    "optimize",
    "process",
    "project",
    "rank",
    "reduce",
    "return",
    "run",
    "score",
    "segment",
    "select",
    "summarize",
    "transform",
    "update",
    "validate",
}
COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "with",
}
CANONICAL_BANNED_TOKENS = {
    "ecg",
    "ppg",
    "eeg",
    "emg",
    "pcg",
    "bpm",
    "beat",
    "heart",
    "rr",
    "qrs",
    "sqi",
    "baseline",
    "wander",
}


def _split_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {token for token in re.split(r"[^A-Za-z0-9]+", expanded.lower()) if token}


def _record_tokens(record: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    tokens |= _split_tokens(record.get("atom_name"))
    tokens |= _split_tokens(record.get("atom_key"))
    tokens |= _split_tokens(record.get("wrapper_symbol"))
    tokens |= _split_tokens(record.get("module_path"))
    tokens |= _split_tokens(record.get("domain_family"))
    tokens |= _split_tokens(record.get("module_family"))
    return tokens


def evaluate_heuristic_asset(record: dict[str, Any]) -> dict[str, Any]:
    """Assess whether a heuristic asset is sufficiently dejargonized."""
    source_kind = record.get("source_kind")
    if source_kind not in HEURISTIC_SOURCE_KINDS:
        return {
            "status": "not_applicable",
            "findings": [],
            "notes": [],
            "plain_language_tokens": [],
            "summary_tokens": [],
        }

    summary = (record.get("docstring_summary") or "").strip()
    summary_lower = summary.lower()
    summary_tokens = sorted(_split_tokens(summary))
    findings: list[str] = []
    notes: list[str] = []

    if not summary:
        return {
            "status": "fail",
            "findings": ["HEURISTIC_DOCSTRING_MISSING"],
            "notes": ["Heuristic assets need a plain-language docstring summary."],
            "plain_language_tokens": [],
            "summary_tokens": [],
        }

    if any(pattern in summary_lower for pattern in PLACEHOLDER_DOC_PATTERNS):
        return {
            "status": "fail",
            "findings": ["HEURISTIC_DOCSTRING_PLACEHOLDER"],
            "notes": ["Docstring summary appears placeholder-like for a heuristic asset."],
            "plain_language_tokens": [],
            "summary_tokens": summary_tokens,
        }

    context_tokens = _record_tokens(record)
    plain_language_tokens = [
        token
        for token in summary_tokens
        if token not in context_tokens and token not in COMMON_STOPWORDS and not token.isdigit()
    ]
    action_verbs = sorted(token for token in summary_tokens if token in ACTION_VERBS)
    acronym_tokens = sorted(token for token in summary_tokens if token.isupper() and len(token) >= 3)

    if not action_verbs:
        findings.append("HEURISTIC_ACTION_VERB_MISSING")
        notes.append("Docstring summary should start with a plain-language action verb.")
    if len(plain_language_tokens) < 3:
        findings.append("HEURISTIC_DEJARGONIZATION_WEAK")
        notes.append("Docstring summary should explain the behavior beyond wrapper/module identifiers.")
    if acronym_tokens and len(plain_language_tokens) < 4:
        findings.append("HEURISTIC_ACRONYM_HEAVY")
        notes.append("Docstring summary relies on acronyms without enough explanatory context.")
    if len(summary_tokens) <= 2:
        findings.append("HEURISTIC_DOCSTRING_THIN")
        notes.append("Docstring summary is very short for a heuristic asset.")

    status = "pass" if not findings else "partial"
    return {
        "status": status,
        "findings": sorted(set(findings)),
        "notes": notes,
        "plain_language_tokens": plain_language_tokens,
        "summary_tokens": summary_tokens,
    }


def _text_contains_banned_shared_token(text: str | None) -> bool:
    tokens = _split_tokens(text)
    return any(token in CANONICAL_BANNED_TOKENS for token in tokens)


def validate_canonical_heuristic_asset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for entry in payload.get("heuristics", []) or []:
        heuristic_id = str(entry.get("heuristic_id", "") or "")
        display_name = str(entry.get("display_name", "") or "")
        meaning = str(entry.get("dejargonized_meaning", "") or "")
        if _text_contains_banned_shared_token(heuristic_id):
            findings.append(
                {
                    "level": "error",
                    "code": "HEURISTIC_CANONICAL_ID_JARGON",
                    "heuristic_id": heuristic_id,
                    "message": "Canonical heuristic identifiers must remain de-jargonized.",
                }
            )
        if _text_contains_banned_shared_token(display_name):
            findings.append(
                {
                    "level": "warning",
                    "code": "HEURISTIC_CANONICAL_DISPLAY_JARGON",
                    "heuristic_id": heuristic_id,
                    "message": "Canonical heuristic display names should avoid domain-specific jargon.",
                }
            )
        if _text_contains_banned_shared_token(meaning):
            findings.append(
                {
                    "level": "warning",
                    "code": "HEURISTIC_CANONICAL_MEANING_JARGON",
                    "heuristic_id": heuristic_id,
                    "message": "Canonical heuristic meaning should stay portable across families.",
                }
            )
    return findings


def validate_family_heuristic_registry_asset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for binding in payload.get("heuristic_bindings", []) or []:
        heuristic_id = str(binding.get("heuristic_id", "") or "")
        if binding.get("display_name") or binding.get("dejargonized_meaning"):
            findings.append(
                {
                    "level": "error",
                    "code": "HEURISTIC_FAMILY_REDEFINITION",
                    "heuristic_id": heuristic_id,
                    "message": "Family heuristic registries may not redefine canonical heuristic fields.",
                }
            )
        if not binding.get("family_notes"):
            findings.append(
                {
                    "level": "error",
                    "code": "HEURISTIC_FAMILY_NOTES_MISSING",
                    "heuristic_id": heuristic_id,
                    "message": "Family heuristic bindings must explain local interpretation.",
                }
            )
    return findings


def validate_atom_heuristic_metadata_asset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    atom_fqdn = str(payload.get("atom_fqdn", "") or "")
    for output in payload.get("heuristic_outputs", []) or []:
        heuristic = output.get("heuristic", {}) if isinstance(output, dict) else {}
        heuristic_id = str(heuristic.get("heuristic_id", "") or "")
        scope = str(heuristic.get("applicability_scope", "") or "")
        meaning = str(heuristic.get("dejargonized_meaning", "") or "")
        if scope == "cross_family" and _text_contains_banned_shared_token(meaning):
            findings.append(
                {
                    "level": "warning",
                    "code": "HEURISTIC_METADATA_CANONICAL_MEANING_JARGON",
                    "heuristic_id": heuristic_id,
                    "atom_fqdn": atom_fqdn,
                    "message": "Cross-family heuristic meaning embedded in atom metadata should remain de-jargonized.",
                }
            )
    return findings
