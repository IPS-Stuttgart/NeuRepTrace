"""Patch OpenNeuro diagnostics manifest provenance normalization."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _is_empty_value(value: Any) -> bool:
    """Return whether a manifest/provenance value should be treated as absent."""

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict | list | tuple):
        return len(value) == 0
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool) or getattr(missing, "shape", None) == ():
        return bool(missing)
    return False


def _normalize_manifest_provenance_value(value: Any) -> Any:
    """Return a CSV-stable representation for non-scalar manifest values."""

    if isinstance(value, list | tuple):
        return "|".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def install() -> None:
    """Install manifest provenance guards into ``openneuro_decode_diagnostics``."""

    from neureptrace import openneuro_decode_diagnostics as diagnostics

    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if _is_empty_value(value):
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _optional_unique_bool(value: Any, *, column: str) -> bool | None:
        if _is_empty_value(value):
            return None
        tokens = [
            token.strip()
            for token in str(value).replace(",", "|").split("|")
            if token.strip()
        ]
        if not tokens:
            return None
        parsed = {diagnostics._parse_bool_token(token) for token in tokens}
        if len(parsed) != 1:
            raise ValueError(f"Inconsistent boolean provenance for {column!r}: {tokens}")
        return parsed.pop()

    def _provenance_value(
        manifest: dict[str, Any],
        summary_provenance: dict[str, str],
        manifest_key: str,
        summary_key: str | None = None,
    ) -> Any:
        value = manifest.get(manifest_key, "")
        if not _is_empty_value(value):
            return _normalize_manifest_provenance_value(value)
        return summary_provenance.get(summary_key or manifest_key, "")

    def _diagnostics_best_time(output_dirs, explicit_best_time: float | None) -> float | None:
        if explicit_best_time is not None:
            return explicit_best_time
        for output_dir in output_dirs:
            value = diagnostics._read_json(output_dir / "run_manifest.json").get("diagnostics_best_time", "")
            if not _is_empty_value(value):
                numeric = diagnostics._as_float(value)
                if numeric is not None:
                    return numeric
        return None

    diagnostics._as_bool = _as_bool
    diagnostics._optional_unique_bool = _optional_unique_bool
    diagnostics._provenance_value = _provenance_value
    diagnostics._diagnostics_best_time = _diagnostics_best_time
