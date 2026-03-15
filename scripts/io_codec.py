"""Serialize and deserialize atom I/O for fixture capture.

Handles numpy arrays, scalars, nested containers, Pydantic models,
and falls back to pickle for opaque types.
"""
from __future__ import annotations

import base64
import functools
import inspect
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_NDARRAY_TAG = "__ndarray__"
_BYTES_TAG = "__bytes__"
_TUPLE_TAG = "__tuple__"
_COMPLEX_TAG = "__complex__"
_PICKLE_TAG = "__pickle__"
_PYDANTIC_TAG = "__pydantic__"

MAX_SERIALIZED_BYTES = 1_000_000  # 1 MB cap per value


def serialize_value(obj: Any) -> Any:
    """Convert *obj* to a JSON-safe representation."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, complex):
        return {_COMPLEX_TAG: True, "real": obj.real, "imag": obj.imag}

    if isinstance(obj, np.generic):
        return serialize_value(obj.item())

    if isinstance(obj, np.ndarray):
        raw = obj.tobytes()
        if len(raw) > MAX_SERIALIZED_BYTES:
            warnings.warn(
                f"ndarray too large ({len(raw)} bytes), truncating fixture",
                stacklevel=2,
            )
            return None
        return {
            _NDARRAY_TAG: True,
            "data": base64.b64encode(raw).decode("ascii"),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }

    if isinstance(obj, bytes):
        if len(obj) > MAX_SERIALIZED_BYTES:
            return None
        return {_BYTES_TAG: True, "data": base64.b64encode(obj).decode("ascii")}

    if isinstance(obj, tuple):
        return {_TUPLE_TAG: True, "items": [serialize_value(v) for v in obj]}

    if isinstance(obj, list):
        return [serialize_value(v) for v in obj]

    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}

    # Pydantic v2 models
    if hasattr(obj, "model_dump"):
        return {
            _PYDANTIC_TAG: True,
            "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "data": serialize_value(obj.model_dump()),
        }

    # Pickle fallback
    try:
        raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if len(raw) > MAX_SERIALIZED_BYTES:
            warnings.warn(
                f"Pickle of {type(obj).__name__} too large, skipping", stacklevel=2
            )
            return None
        warnings.warn(
            f"Falling back to pickle for {type(obj).__name__}", stacklevel=2
        )
        return {
            _PICKLE_TAG: True,
            "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "data": base64.b64encode(raw).decode("ascii"),
        }
    except Exception:
        warnings.warn(f"Cannot serialize {type(obj).__name__}, skipping", stacklevel=2)
        return None


def serialize_args(
    args: tuple, kwargs: dict, func: Any | None = None
) -> dict[str, Any]:
    """Map positional + keyword arguments to a JSON-safe dict.

    If *func* is provided, positional args are mapped to parameter names
    via ``inspect.signature``.
    """
    named: dict[str, Any] = {}
    if func is not None:
        try:
            params = list(inspect.signature(func).parameters.keys())
        except (ValueError, TypeError):
            params = []
    else:
        params = []

    for i, val in enumerate(args):
        key = params[i] if i < len(params) else f"_arg{i}"
        named[key] = serialize_value(val)

    for k, v in kwargs.items():
        named[k] = serialize_value(v)

    return named


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def deserialize_value(obj: Any) -> Any:
    """Reconstruct a Python object from its JSON-safe representation."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, list):
        return [deserialize_value(v) for v in obj]

    if isinstance(obj, dict):
        if obj.get(_NDARRAY_TAG):
            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(
                obj["shape"]
            )

        if obj.get(_BYTES_TAG):
            return base64.b64decode(obj["data"])

        if obj.get(_TUPLE_TAG):
            return tuple(deserialize_value(v) for v in obj["items"])

        if obj.get(_COMPLEX_TAG):
            return complex(obj["real"], obj["imag"])

        if obj.get(_PICKLE_TAG):
            return pickle.loads(base64.b64decode(obj["data"]))

        if obj.get(_PYDANTIC_TAG):
            # Return the raw dict — caller can reconstruct if needed
            return deserialize_value(obj["data"])

        return {k: deserialize_value(v) for k, v in obj.items()}

    return obj


# ---------------------------------------------------------------------------
# Fixture I/O
# ---------------------------------------------------------------------------


def save_fixture(records: list[dict], path: str | Path) -> None:
    """Write a list of call records to a JSON fixture file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, allow_nan=False)


def load_fixture(path: str | Path) -> list[dict]:
    """Load call records from a JSON fixture file."""
    with open(path) as f:
        return json.load(f)


def deserialize_inputs(raw: dict) -> dict:
    """Deserialize a fixture's ``inputs`` dict back to Python objects."""
    return {k: deserialize_value(v) for k, v in raw.items()}


def deserialize_output(raw: Any) -> Any:
    """Deserialize a fixture's ``output`` value."""
    return deserialize_value(raw)
