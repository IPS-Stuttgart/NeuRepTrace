"""Reject invalid reaction-time numeric configuration values."""

from __future__ import annotations

import importlib
from dataclasses import replace
from numbers import Integral
from typing import Any

import numpy as np

_INSTALLED = False


def _is_boolean_scalar(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _scale_error(value: Any) -> ValueError:
    return ValueError(f"reaction_time_scale must be a finite numeric scale, got {value!r}.")


def _validate_reaction_time_scale(reaction_time_scale: Any) -> float:
    if _is_boolean_scalar(reaction_time_scale):
        raise _scale_error(reaction_time_scale)
    try:
        scale = float(reaction_time_scale)
    except (TypeError, ValueError) as exc:
        raise _scale_error(reaction_time_scale) from exc
    if not np.isfinite(scale):
        raise _scale_error(reaction_time_scale)
    return scale


def _validate_trial_index_base(trial_index_base: Any) -> int:
    module = importlib.import_module("neureptrace.behavior.reaction_time")
    choices = getattr(module, "TRIAL_INDEX_BASE_CHOICES", (0, 1))
    if _is_boolean_scalar(trial_index_base) or not isinstance(trial_index_base, Integral) or trial_index_base not in choices:
        raise ValueError(f"trial_index_base must be one of {choices}, got {trial_index_base!r}.")
    return int(trial_index_base)


def install() -> None:
    """Install stricter validation into the reaction-time helpers."""

    global _INSTALLED
    if _INSTALLED:
        return
    module = importlib.import_module("neureptrace.behavior.reaction_time")
    original_load_reaction_time_csv = module.load_reaction_time_csv
    original_reaction_time_rows_from_values = module.reaction_time_rows_from_values
    original_extract_reaction_times_from_metadata = module.extract_reaction_times_from_metadata

    def load_reaction_time_csv(path, config=None):
        if config is not None:
            scale = _validate_reaction_time_scale(config.reaction_time_scale)
            config = replace(config, reaction_time_scale=scale)
        return original_load_reaction_time_csv(path, config)

    def reaction_time_rows_from_values(
        values,
        *,
        participant=None,
        dataset="main",
        reaction_time_scale=1.0,
    ):
        return original_reaction_time_rows_from_values(
            values,
            participant=participant,
            dataset=dataset,
            reaction_time_scale=_validate_reaction_time_scale(reaction_time_scale),
        )

    def extract_reaction_times_from_metadata(
        metadata,
        *,
        reaction_time_column=None,
        participant=None,
        dataset="main",
        reaction_time_scale=1.0,
    ):
        return original_extract_reaction_times_from_metadata(
            metadata,
            reaction_time_column=reaction_time_column,
            participant=participant,
            dataset=dataset,
            reaction_time_scale=_validate_reaction_time_scale(reaction_time_scale),
        )

    module._validate_trial_index_base = _validate_trial_index_base
    module._validate_reaction_time_scale = _validate_reaction_time_scale
    module.load_reaction_time_csv = load_reaction_time_csv
    module.reaction_time_rows_from_values = reaction_time_rows_from_values
    module.extract_reaction_times_from_metadata = extract_reaction_times_from_metadata
    _INSTALLED = True
