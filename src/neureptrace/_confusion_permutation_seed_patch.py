"""Runtime patch for confusion-category permutation seed validation."""

from __future__ import annotations

import numpy as np

_ERROR_MESSAGE = "seed must be a non-negative integer or None."


def _validate_seed(seed: int | None) -> int:
    if seed is None:
        return 0
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError(_ERROR_MESSAGE)
    try:
        numeric = float(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0.0:
        raise ValueError(_ERROR_MESSAGE)
    return int(numeric)


def install() -> None:
    """Install a seed validator for confusion-category permutation tests."""
    import neureptrace.metrics.confusion as confusion_metrics

    if getattr(confusion_metrics._category_seed, "_validated_seed_patched", False):
        return

    def _category_seed(seed: int | None, group_values: dict[str, object], category_column: str):
        seed_values = [_validate_seed(seed), sum(ord(character) for character in str(category_column))]
        for key, value in sorted(group_values.items()):
            seed_values.append(sum(ord(character) for character in f"{key}={value}"))
        return np.random.SeedSequence(seed_values)

    _category_seed._validated_seed_patched = True  # type: ignore[attr-defined]
    confusion_metrics._category_seed = _category_seed


__all__ = ["install"]
