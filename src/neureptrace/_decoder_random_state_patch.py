"""Propagate decoder factory random_state into nested sklearn estimators."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_decoder_random_state_patch_installed"


def _nested_random_state_updates(estimator: Any, random_state: int | None) -> dict[str, int | None]:
    """Return set_params updates for every nested random_state parameter."""

    if not (hasattr(estimator, "get_params") and hasattr(estimator, "set_params")):
        return {}
    try:
        params = estimator.get_params(deep=True)
    except Exception:  # pragma: no cover - third-party estimator defensive guard
        return {}
    return {
        name: random_state
        for name in params
        if name == "random_state" or name.endswith("__random_state")
    }


def _propagate_random_state(estimator: Any, random_state: int | None) -> Any:
    updates = _nested_random_state_updates(estimator, random_state)
    if updates:
        estimator.set_params(**updates)
    return estimator


def install() -> None:
    """Patch decoder factories so their random_state argument is honored deeply."""

    from neureptrace import decoding

    if getattr(decoding, _PATCH_MARKER, False):
        return

    original_make_decoder = decoding.make_decoder
    original_make_tuned_decoder = decoding.make_tuned_decoder

    @wraps(original_make_decoder)
    def make_decoder(*args: Any, **kwargs: Any):
        random_state = kwargs.get("random_state", 13)
        estimator = original_make_decoder(*args, **kwargs)
        return _propagate_random_state(estimator, random_state)

    @wraps(original_make_tuned_decoder)
    def make_tuned_decoder(*args: Any, **kwargs: Any):
        random_state = kwargs.get("random_state", 13)
        estimator = original_make_tuned_decoder(*args, **kwargs)
        return _propagate_random_state(estimator, random_state)

    decoding.make_decoder = make_decoder
    decoding.make_tuned_decoder = make_tuned_decoder
    setattr(decoding, _PATCH_MARKER, True)


__all__ = ["install"]
