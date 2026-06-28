"""Make OpenNeuro diagnostics boolean provenance parsing robust to list values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

import pandas as pd

_PATCH_MARKER = "_neureptrace_openneuro_decode_diagnostics_scalar_bool_patch_installed"
_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (Mapping, Iterable)) and not isinstance(value, (str, bytes)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _tokenize_bool_values(value: Any) -> list[Any]:
    if _is_missing_scalar(value):
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.replace(",", "|").split("|") if token.strip()]
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
        return [token.strip() for token in text.replace(",", "|").split("|") if token.strip()]
    if isinstance(value, Mapping):
        raise ValueError("Boolean provenance values must be scalar or list-like, not mappings.")
    if isinstance(value, Iterable):
        tokens: list[Any] = []
        for item in value:
            tokens.extend(_tokenize_bool_values(item))
        return tokens
    return [value]


def _parse_bool_token(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    raise ValueError(f"Cannot parse boolean provenance value {value!r}.")


def _as_bool(value: Any) -> bool:
    tokens = _tokenize_bool_values(value)
    if not tokens:
        return False
    parsed = {_parse_bool_token(token) for token in tokens}
    if len(parsed) != 1:
        raise ValueError(f"Inconsistent boolean provenance value: {tokens}")
    return parsed.pop()


def _optional_unique_bool(value: Any, *, column: str) -> bool | None:
    tokens = _tokenize_bool_values(value)
    if not tokens:
        return None
    parsed = {_parse_bool_token(token) for token in tokens}
    if len(parsed) != 1:
        raise ValueError(f"Inconsistent boolean provenance for {column!r}: {tokens}")
    return parsed.pop()


def install() -> None:
    """Patch OpenNeuro diagnostics helpers that previously assumed scalar values."""

    import neureptrace.openneuro_decode_diagnostics as diagnostics

    original_provenance_value = diagnostics._provenance_value
    if getattr(original_provenance_value, _PATCH_MARKER, False):
        return

    @wraps(original_provenance_value)
    def _provenance_value(
        manifest: dict[str, Any],
        summary_provenance: dict[str, str],
        manifest_key: str,
        summary_key: str | None = None,
    ) -> Any:
        value = manifest.get(manifest_key, "")
        if not _is_missing_scalar(value):
            return value
        return summary_provenance.get(summary_key or manifest_key, "")

    setattr(_provenance_value, _PATCH_MARKER, True)
    diagnostics._parse_bool_token = _parse_bool_token
    diagnostics._as_bool = _as_bool
    diagnostics._optional_unique_bool = _optional_unique_bool
    diagnostics._provenance_value = _provenance_value


__all__ = ["install"]
