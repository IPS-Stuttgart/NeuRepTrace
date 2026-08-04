"""Runtime patch for sign-flip scalar controls and robust statistics."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from numbers import Integral, Real

import numpy as np

_PERMUTATION_COUNT_ERROR = "n_permutations must be a positive integer."
_RANDOM_STATE_ERROR = "random_state must be a non-negative integer seed."
_CLUSTER_ALPHA_ERROR = "cluster_alpha must be between 0 and 1."
_REFERENCE_VALUE_ERROR = "chance must be a finite numeric scalar."
_PATCH_MARKER = "_sign_flip_scalar_controls_patched"
_REFERENCE_VALUE_PATCH_MARKER = "_sign_flip_reference_value_patched"
_T_STATISTIC_PATCH_MARKER = "_sign_flip_zero_variance_t_patched"


class _NanSafeQuantileNumpyProxy:
    """Delegate NumPy operations while avoiding NaN quantiles over infinities."""

    def __init__(self, numpy_module: object) -> None:
        self._numpy = numpy_module

    def __getattr__(self, name: str) -> object:
        return getattr(self._numpy, name)

    def quantile(self, values: object, quantile: object, *args: object, **kwargs: object) -> object:
        """Use the existing interpolation unless infinities make it return NaN."""

        with self._numpy.errstate(invalid="ignore"):
            threshold = self._numpy.quantile(values, quantile, *args, **kwargs)
        if not self._numpy.isnan(threshold).any():
            return threshold

        fallback_args = list(args)
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("interpolation", None)
        if len(fallback_args) >= 4:
            fallback_args[3] = "higher"
            fallback_kwargs.pop("method", None)
        else:
            fallback_kwargs["method"] = "higher"
        return self._numpy.quantile(values, quantile, *fallback_args, **fallback_kwargs)


def _scalar_value(value: object, error_message: str) -> object:
    """Return a zero-dimensional scalar value, rejecting arrays and booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(error_message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc
    if array.ndim != 0:
        raise ValueError(error_message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(error_message)
    return scalar


def _scalar_float(value: object, error_message: str) -> float:
    """Return a float from a scalar-like numeric value, rejecting arrays and booleans."""
    scalar = _scalar_value(value, error_message)
    try:
        return float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc


def _decimal_integer(value: object, error_message: str) -> int:
    """Parse an exact integral decimal representation without a float round-trip."""
    if isinstance(value, bytes):
        try:
            text = value.decode().strip()
        except UnicodeDecodeError as exc:
            raise ValueError(error_message) from exc
    else:
        text = str(value).strip()
    if not text:
        raise ValueError(error_message)
    try:
        numeric = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(error_message) from exc
    if not numeric.is_finite():
        raise ValueError(error_message)
    integral = numeric.to_integral_value()
    if numeric != integral:
        raise ValueError(error_message)
    return int(integral)


def _scalar_integer(value: object, error_message: str) -> int:
    """Return an exact integer from an integer or integral scalar representation."""
    scalar = _scalar_value(value, error_message)
    if isinstance(scalar, Integral):
        return int(scalar)
    if isinstance(scalar, Decimal):
        if not scalar.is_finite():
            raise ValueError(error_message)
        integral = scalar.to_integral_value()
        if scalar != integral:
            raise ValueError(error_message)
        return int(integral)
    if isinstance(scalar, Real):
        numeric = float(scalar)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(error_message)
        return int(numeric)
    return _decimal_integer(scalar, error_message)


def _validate_positive_permutation_count(n_permutations: int) -> int:
    integer = _scalar_integer(n_permutations, _PERMUTATION_COUNT_ERROR)
    if integer < 1:
        raise ValueError(_PERMUTATION_COUNT_ERROR)
    return integer


def _validate_random_state(random_state: int) -> int:
    integer = _scalar_integer(random_state, _RANDOM_STATE_ERROR)
    if integer < 0:
        raise ValueError(_RANDOM_STATE_ERROR)
    return integer


def _validate_cluster_alpha(cluster_alpha: float) -> float:
    numeric = _scalar_float(cluster_alpha, _CLUSTER_ALPHA_ERROR)
    if not np.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(_CLUSTER_ALPHA_ERROR)
    return float(numeric)


def _validate_reference_value(reference_value: object) -> float:
    numeric = _scalar_float(reference_value, _REFERENCE_VALUE_ERROR)
    if not np.isfinite(numeric):
        raise ValueError(_REFERENCE_VALUE_ERROR)
    return float(numeric)


def _t_statistics_from_mean_and_sem(means: np.ndarray, sem: np.ndarray) -> np.ndarray:
    """Return t statistics, including signed infinities for exact nonzero constants."""

    statistics = np.divide(
        means,
        sem,
        out=np.zeros_like(means, dtype=float),
        where=sem > 0.0,
    )
    zero_variance = sem == 0.0
    statistics[zero_variance & (means > 0.0)] = np.inf
    statistics[zero_variance & (means < 0.0)] = -np.inf
    return statistics


def _scale_effect_columns(effects: np.ndarray) -> np.ndarray:
    """Scale finite effect columns without changing their t statistics."""

    if not np.isfinite(effects).all():
        raise ValueError("Subject-level effects must contain only finite values.")
    scales = np.max(np.abs(effects), axis=0)
    safe_scales = np.where(scales > 0.0, scales, 1.0)
    return effects / safe_scales


def _t_statistic(effects: np.ndarray) -> np.ndarray:
    """Compute overflow-safe one-sample t statistics for finite effects."""

    if effects.shape[0] < 2:
        raise ValueError("Need at least two subjects for subject-level inference.")
    scaled_effects = _scale_effect_columns(effects)
    means = scaled_effects.mean(axis=0)
    sem = scaled_effects.std(axis=0, ddof=1) / np.sqrt(effects.shape[0])
    return _t_statistics_from_mean_and_sem(means, sem)


def _patch_inference() -> None:
    import neureptrace.inference as inference

    if not isinstance(inference.np, _NanSafeQuantileNumpyProxy):
        inference.np = _NanSafeQuantileNumpyProxy(inference.np)

    if not getattr(inference._validate_positive_permutation_count, _PATCH_MARKER, False):
        _validate_positive_permutation_count._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        inference._validate_positive_permutation_count = _validate_positive_permutation_count

    if not getattr(inference._t_statistic, _T_STATISTIC_PATCH_MARKER, False):
        _t_statistic._sign_flip_zero_variance_t_patched = True  # type: ignore[attr-defined]
        inference._t_statistic = _t_statistic

    if not getattr(inference._sign_flip_t_statistics, _PATCH_MARKER, False):
        original_sign_flip_t_statistics = inference._sign_flip_t_statistics

        @wraps(original_sign_flip_t_statistics)
        def _sign_flip_t_statistics(
            effects: np.ndarray,
            *,
            n_permutations: int,
            random_state: int,
        ) -> np.ndarray:
            validated_permutations = _validate_positive_permutation_count(n_permutations)
            validated_random_state = _validate_random_state(random_state)
            rng = np.random.default_rng(validated_random_state)
            n_subjects = effects.shape[0]
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(validated_permutations, n_subjects),
            )
            scaled_effects = _scale_effect_columns(effects)
            means = signs @ scaled_effects / n_subjects
            sum_squares = np.sum(scaled_effects**2, axis=0)
            variances = (sum_squares[None, :] - n_subjects * means**2) / (n_subjects - 1)
            sem = np.sqrt(np.maximum(variances, 0.0) / n_subjects)
            return _t_statistics_from_mean_and_sem(means, sem)

        _sign_flip_t_statistics._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        inference._sign_flip_t_statistics = _sign_flip_t_statistics

    if not getattr(inference.subject_time_effects, _REFERENCE_VALUE_PATCH_MARKER, False):
        original_subject_time_effects = inference.subject_time_effects

        @wraps(original_subject_time_effects)
        def subject_time_effects(*args: object, **kwargs: object):
            validated_kwargs = dict(kwargs)
            if "chance" in validated_kwargs:
                validated_kwargs["chance"] = _validate_reference_value(validated_kwargs["chance"])
            return original_subject_time_effects(*args, **validated_kwargs)

        subject_time_effects._sign_flip_reference_value_patched = True  # type: ignore[attr-defined]
        inference.subject_time_effects = subject_time_effects

    if not getattr(inference.sign_flip_time_inference, _PATCH_MARKER, False):
        original_sign_flip_time_inference = inference.sign_flip_time_inference

        @wraps(original_sign_flip_time_inference)
        def sign_flip_time_inference(*args: object, **kwargs: object):
            validated_kwargs = dict(kwargs)
            if "chance" in validated_kwargs:
                validated_kwargs["chance"] = _validate_reference_value(validated_kwargs["chance"])
            if "n_permutations" in validated_kwargs:
                validated_kwargs["n_permutations"] = _validate_positive_permutation_count(validated_kwargs["n_permutations"])
            if "random_state" in validated_kwargs:
                validated_kwargs["random_state"] = _validate_random_state(validated_kwargs["random_state"])
            if "cluster_alpha" in validated_kwargs:
                validated_kwargs["cluster_alpha"] = _validate_cluster_alpha(validated_kwargs["cluster_alpha"])
            return original_sign_flip_time_inference(*args, **validated_kwargs)

        sign_flip_time_inference._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        inference.sign_flip_time_inference = sign_flip_time_inference


def _patch_paired_stats() -> None:
    import neureptrace.paired_stats as paired_stats

    if not getattr(paired_stats._validate_positive_permutation_count, _PATCH_MARKER, False):
        _validate_positive_permutation_count._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        paired_stats._validate_positive_permutation_count = _validate_positive_permutation_count
    if not getattr(paired_stats._validate_random_state, _PATCH_MARKER, False):
        _validate_random_state._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        paired_stats._validate_random_state = _validate_random_state


def install() -> None:
    """Install strict scalar controls and overflow-safe sign-flip statistics."""
    _patch_inference()
    _patch_paired_stats()


__all__ = ["install"]
