"""Runtime patch for confusion-category permutation scalar validation."""

from __future__ import annotations

import numpy as np

_SEED_ERROR_MESSAGE = "seed must be a non-negative integer or None."
_PERMUTATION_COUNT_ERROR_MESSAGE = "n_permutations must be a non-negative integer."


def _coerce_non_negative_integer(value: object, *, error_message: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(error_message)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            raise ValueError(error_message)
        if value.ndim != 0:
            raise ValueError(error_message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(error_message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0.0:
        raise ValueError(error_message)
    return int(numeric)


def _validate_seed(seed: int | None) -> int:
    if seed is None:
        return 0
    return _coerce_non_negative_integer(seed, error_message=_SEED_ERROR_MESSAGE)


def _validate_permutation_count(n_permutations: int | None) -> int | None:
    if n_permutations is None:
        return None
    return _coerce_non_negative_integer(n_permutations, error_message=_PERMUTATION_COUNT_ERROR_MESSAGE)


def install() -> None:
    """Install validators for confusion-category permutation controls."""
    import neureptrace.metrics.confusion as confusion_metrics

    if not getattr(confusion_metrics._category_seed, "_validated_seed_patched", False):

        def _category_seed(seed: int | None, group_values: dict[str, object], category_column: str):
            seed_values = [_validate_seed(seed), sum(ord(character) for character in str(category_column))]
            for key, value in sorted(group_values.items()):
                seed_values.append(sum(ord(character) for character in f"{key}={value}"))
            return np.random.SeedSequence(seed_values)

        _category_seed._validated_seed_patched = True  # type: ignore[attr-defined]
        confusion_metrics._category_seed = _category_seed

    if not getattr(confusion_metrics._validate_permutation_count, "_validated_permutation_count_patched", False):
        _validate_permutation_count._validated_permutation_count_patched = True  # type: ignore[attr-defined]
        confusion_metrics._validate_permutation_count = _validate_permutation_count


__all__ = ["install"]
