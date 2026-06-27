"""Reject duplicate or alias-equivalent ensemble source decoders."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any


def _duplicate_exact_values(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _decoder_aliases(name: str) -> tuple[str, ...]:
    text = str(name)
    return tuple(dict.fromkeys((text, text.replace("-", "_"), text.replace("_", "-"))))


def _duplicate_alias_pairs(decoders: Sequence[Any]) -> tuple[str, ...]:
    alias_owner: dict[str, str] = {}
    duplicate_pairs: list[str] = []
    for decoder in decoders:
        name = str(decoder)
        aliases = _decoder_aliases(name)
        owner = next((alias_owner[alias] for alias in aliases if alias in alias_owner), None)
        if owner is not None:
            pair = f"{owner!r}/{name!r}"
            if pair not in duplicate_pairs:
                duplicate_pairs.append(pair)
        for alias in aliases:
            alias_owner.setdefault(alias, name)
    return tuple(duplicate_pairs)


def _format_duplicates(duplicates: Sequence[str]) -> str:
    return ", ".join(str(value) for value in duplicates)


def install() -> None:
    """Install duplicate-source validation for probability ensembling."""

    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    original_observation_ensemble = observation_ensemble.ensemble_probability_observations

    def ensemble_probability_observations(observations, *args, **kwargs):
        decoders = kwargs.get("decoders", observation_ensemble.DEFAULT_DECODERS)
        if decoders is not None:
            duplicate_pairs = _duplicate_alias_pairs(tuple(decoders))
            if duplicate_pairs:
                raise ValueError(
                    "Ensemble source decoders must be unique after alias normalization; "
                    f"duplicates: {_format_duplicates(duplicate_pairs)}."
                )
        return original_observation_ensemble(observations, *args, **kwargs)

    observation_ensemble.ensemble_probability_observations = ensemble_probability_observations

    time_decode_ensemble = importlib.import_module("neureptrace.mne_time_decode_ensemble")
    original_parse_source_decoders = time_decode_ensemble._parse_source_decoders

    def _parse_source_decoders(source_decoders):
        requests, normalized = original_parse_source_decoders(source_decoders)
        duplicates = _duplicate_exact_values(tuple(str(decoder) for decoder in normalized))
        if duplicates:
            raise ValueError(
                "logistic_svm_ensemble source decoders must be unique after alias normalization; "
                f"duplicates: {_format_duplicates(duplicates)}."
            )
        return requests, normalized

    time_decode_ensemble._parse_source_decoders = _parse_source_decoders


__all__ = ["install"]
