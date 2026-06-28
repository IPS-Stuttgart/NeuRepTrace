"""Reject non-string values in decoder option normalizers."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_decoding_option_type_validation_patch_installed"


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def _require_string_or_none(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None.")
    return value


def install() -> None:
    """Install polite type validation for public decoding option normalizers."""

    decoding = importlib.import_module("neureptrace.decoding")
    if getattr(decoding, _PATCH_MARKER, False):
        return

    original_normalize_decoder_name = decoding.normalize_decoder_name
    original_normalize_emission_mode = decoding.normalize_emission_mode
    original_normalize_feature_preprocessor = decoding.normalize_feature_preprocessor
    original_normalize_tuning_scoring = decoding.normalize_tuning_scoring

    @wraps(original_normalize_decoder_name)
    def normalize_decoder_name(name):
        return original_normalize_decoder_name(_require_string(name, name="decoder"))

    @wraps(original_normalize_emission_mode)
    def normalize_emission_mode(mode):
        return original_normalize_emission_mode(_require_string(mode, name="emission_mode"))

    @wraps(original_normalize_feature_preprocessor)
    def normalize_feature_preprocessor(name):
        return original_normalize_feature_preprocessor(_require_string_or_none(name, name="feature_preprocessor"))

    @wraps(original_normalize_tuning_scoring)
    def normalize_tuning_scoring(scoring):
        return original_normalize_tuning_scoring(_require_string(scoring, name="tuning_scoring"))

    decoding.normalize_decoder_name = normalize_decoder_name
    decoding.normalize_emission_mode = normalize_emission_mode
    decoding.normalize_feature_preprocessor = normalize_feature_preprocessor
    decoding.normalize_tuning_scoring = normalize_tuning_scoring
    setattr(decoding, _PATCH_MARKER, True)


__all__ = ["install"]
