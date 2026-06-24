"""Online unlabeled test-time adaptation for probability traces.

The adapter implements a Category 2c protocol: it may use held-out target
features through source-model probability outputs, but it never accepts or uses
held-out target labels.  The update is deliberately post-hoc and estimator
agnostic so it can be applied to sklearn, DANN, or aligned-decoder probabilities.

The default ``after_predict`` timing is deployable online adaptation: trial ``t``
is emitted with the bias learned from earlier target trials, then trial ``t`` is
used as an unlabeled update for future predictions.  ``before_predict`` is also
available for explicit transductive/current-sample adaptation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TTIME_PROTOCOL = "category_2c_online_test_time_entropy_adaptation"
TEST_TIME_ADAPTATION_CHOICES = ("none", "ttime")
TEST_TIME_ADAPTATION_UPDATE_TIMING_CHOICES = ("after_predict", "before_predict")
PROBABILITY_EPSILON = 1e-12
PROBABILITY_SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class TestTimeAdaptationResult:
    """Output from online unlabeled probability adaptation."""

    probabilities: np.ndarray
    metadata: dict[str, object]
    final_bias: np.ndarray
    running_marginal: np.ndarray


def normalize_test_time_adaptation(mode: str | None) -> str:
    """Normalize public names for test-time adaptation modes."""

    normalized = "none" if mode is None else str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "off", "false", "no", "disabled"}:
        return "none"
    if normalized in {"t_time", "test_time", "test_time_adaptation", "online_entropy", "ttime_entropy", "tent", "entropy"}:
        return "ttime"
    if normalized not in TEST_TIME_ADAPTATION_CHOICES:
        raise ValueError(f"Unknown test_time_adaptation '{mode}'. Available modes: {', '.join(TEST_TIME_ADAPTATION_CHOICES)}.")
    return normalized


def normalize_ttime_update_timing(timing: str | None) -> str:
    """Normalize whether unlabeled updates happen before or after emitting a row."""

    normalized = "after_predict" if timing is None else str(timing).strip().lower().replace("-", "_")
    if normalized in {"after", "after_prediction", "future_only", "online_after", "online_after_predict"}:
        return "after_predict"
    if normalized in {"before", "before_prediction", "current", "adapt_current", "online_before", "online_before_predict"}:
        return "before_predict"
    if normalized not in TEST_TIME_ADAPTATION_UPDATE_TIMING_CHOICES:
        raise ValueError(
            "Unknown ttime_update_timing "
            f"{timing!r}. Available values: {', '.join(TEST_TIME_ADAPTATION_UPDATE_TIMING_CHOICES)}."
        )
    return normalized


def adapt_probabilities_online(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    mode: str = "ttime",
    source_prior: Sequence[float] | np.ndarray | None = None,
    learning_rate: float = 0.05,
    entropy_weight: float = 1.0,
    marginal_weight: float = 0.1,
    marginal_momentum: float = 0.05,
    max_updates_per_sample: int = 1,
    update_timing: str = "after_predict",
    temperature: float = 1.0,
    bias_clip: float = 3.0,
) -> TestTimeAdaptationResult:
    """Adapt probability rows online without target labels.

    Parameters
    ----------
    probabilities:
        Source-model probabilities for the held-out target sequence, already
        aligned to the global class order.
    mode:
        ``"none"`` returns the input probabilities with provenance metadata.
        ``"ttime"`` applies conditional-entropy minimization with marginal
        distribution regularization.
    source_prior:
        Optional source-trained class prior in the same class order as
        ``probabilities``.  When omitted, a uniform prior is used.
    learning_rate, entropy_weight, marginal_weight:
        Update coefficients for the class-bias vector.  The entropy term sharpens
        confident target predictions; the marginal term discourages class
        collapse by pulling the running target marginal toward ``source_prior``.
    marginal_momentum:
        Exponential moving-average weight assigned to each target row.
    max_updates_per_sample:
        Number of unlabeled bias updates from each target row.
    update_timing:
        ``"after_predict"`` emits each row before updating from it.  This is the
        most deployment-realistic online mode.  ``"before_predict"`` adapts on
        the current unlabeled row before emitting it and should be reported as
        current-sample/transductive adaptation.
    temperature:
        Optional logit temperature applied before the adaptive bias.
    bias_clip:
        Symmetric clipping bound for the centered class-bias vector.

    Returns
    -------
    TestTimeAdaptationResult
        Adapted probabilities and compact provenance metadata.  No target labels
        are accepted by this function.
    """

    mode = normalize_test_time_adaptation(mode)
    base_probabilities = _validate_probability_rows(probabilities)
    n_rows, n_classes = base_probabilities.shape
    prior = _normalize_prior(source_prior, n_classes=n_classes)
    learning_rate = _normalize_nonnegative_float(learning_rate, name="ttime_learning_rate")
    entropy_weight = _normalize_nonnegative_float(entropy_weight, name="ttime_entropy_weight")
    marginal_weight = _normalize_nonnegative_float(marginal_weight, name="ttime_marginal_weight")
    marginal_momentum = _normalize_unit_interval_float(marginal_momentum, name="ttime_marginal_momentum", include_one=True)
    max_updates_per_sample = _normalize_positive_int(max_updates_per_sample, name="ttime_max_updates_per_sample")
    update_timing = normalize_ttime_update_timing(update_timing)
    temperature = _normalize_positive_float(temperature, name="ttime_temperature")
    bias_clip = _normalize_nonnegative_float(bias_clip, name="ttime_bias_clip")

    if mode == "none":
        metadata = _metadata(
            mode=mode,
            n_rows=n_rows,
            n_classes=n_classes,
            source_prior=prior,
            learning_rate=learning_rate,
            entropy_weight=entropy_weight,
            marginal_weight=marginal_weight,
            marginal_momentum=marginal_momentum,
            max_updates_per_sample=max_updates_per_sample,
            update_timing=update_timing,
            temperature=temperature,
            bias_clip=bias_clip,
            final_bias=np.zeros(n_classes, dtype=float),
            running_marginal=prior,
        )
        return TestTimeAdaptationResult(
            probabilities=base_probabilities.copy(),
            metadata=metadata,
            final_bias=np.zeros(n_classes, dtype=float),
            running_marginal=prior.copy(),
        )

    adapted = np.zeros_like(base_probabilities)
    bias = np.zeros(n_classes, dtype=float)
    running_marginal = prior.copy()

    for row_index, row in enumerate(base_probabilities):
        if update_timing == "before_predict":
            for _ in range(max_updates_per_sample):
                current = _apply_bias(row, bias=bias, temperature=temperature)
                running_marginal = _update_running_marginal(running_marginal, current, momentum=marginal_momentum)
                bias = _updated_bias(
                    bias,
                    current,
                    running_marginal,
                    prior,
                    learning_rate=learning_rate,
                    entropy_weight=entropy_weight,
                    marginal_weight=marginal_weight,
                    bias_clip=bias_clip,
                )
            adapted[row_index] = _apply_bias(row, bias=bias, temperature=temperature)
        else:
            adapted[row_index] = _apply_bias(row, bias=bias, temperature=temperature)
            for _ in range(max_updates_per_sample):
                current = _apply_bias(row, bias=bias, temperature=temperature)
                running_marginal = _update_running_marginal(running_marginal, current, momentum=marginal_momentum)
                bias = _updated_bias(
                    bias,
                    current,
                    running_marginal,
                    prior,
                    learning_rate=learning_rate,
                    entropy_weight=entropy_weight,
                    marginal_weight=marginal_weight,
                    bias_clip=bias_clip,
                )

    adapted = _validate_probability_rows(adapted)
    metadata = _metadata(
        mode=mode,
        n_rows=n_rows,
        n_classes=n_classes,
        source_prior=prior,
        learning_rate=learning_rate,
        entropy_weight=entropy_weight,
        marginal_weight=marginal_weight,
        marginal_momentum=marginal_momentum,
        max_updates_per_sample=max_updates_per_sample,
        update_timing=update_timing,
        temperature=temperature,
        bias_clip=bias_clip,
        final_bias=bias,
        running_marginal=running_marginal,
    )
    return TestTimeAdaptationResult(
        probabilities=adapted,
        metadata=metadata,
        final_bias=bias.copy(),
        running_marginal=running_marginal.copy(),
    )


def _metadata(
    *,
    mode: str,
    n_rows: int,
    n_classes: int,
    source_prior: np.ndarray,
    learning_rate: float,
    entropy_weight: float,
    marginal_weight: float,
    marginal_momentum: float,
    max_updates_per_sample: int,
    update_timing: str,
    temperature: float,
    bias_clip: float,
    final_bias: np.ndarray,
    running_marginal: np.ndarray,
) -> dict[str, object]:
    enabled = mode != "none"
    return {
        "test_time_adaptation": mode,
        "test_time_adaptation_protocol": TTIME_PROTOCOL if enabled else "",
        "test_time_adaptation_uses_target_features": bool(enabled),
        "test_time_adaptation_uses_target_labels": False,
        "test_time_adaptation_online": bool(enabled),
        "test_time_adaptation_objective": "conditional_entropy_plus_marginal_regularization" if enabled else "",
        "test_time_adaptation_update_timing": update_timing if enabled else "",
        "test_time_adaptation_n_target_rows": int(n_rows),
        "test_time_adaptation_n_classes": int(n_classes),
        "ttime_learning_rate": float(learning_rate) if enabled else "",
        "ttime_entropy_weight": float(entropy_weight) if enabled else "",
        "ttime_marginal_weight": float(marginal_weight) if enabled else "",
        "ttime_marginal_momentum": float(marginal_momentum) if enabled else "",
        "ttime_max_updates_per_sample": int(max_updates_per_sample) if enabled else "",
        "ttime_temperature": float(temperature) if enabled else "",
        "ttime_bias_clip": float(bias_clip) if enabled else "",
        "ttime_source_prior": _format_vector(source_prior) if enabled else "",
        "ttime_final_bias": _format_vector(final_bias) if enabled else "",
        "ttime_final_bias_l2": float(np.linalg.norm(final_bias)) if enabled else "",
        "ttime_final_running_marginal": _format_vector(running_marginal) if enabled else "",
    }


def _format_vector(values: np.ndarray) -> str:
    return "|".join(f"{float(value):.12g}" for value in np.asarray(values, dtype=float).reshape(-1))


def _validate_probability_rows(probabilities: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must contain at least one row and at least two classes.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must be finite.")
    if np.any(matrix < 0.0):
        raise ValueError("probabilities must be non-negative.")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    normalized = matrix / row_sums
    bad_rows = np.flatnonzero(np.abs(row_sums.reshape(-1) - 1.0) > PROBABILITY_SUM_TOLERANCE)
    if len(bad_rows):
        examples = [float(row_sums.reshape(-1)[index]) for index in bad_rows[:5]]
        raise ValueError(
            "probability rows must sum to 1.0 within tolerance "
            f"{PROBABILITY_SUM_TOLERANCE:g}; example row sums: {examples}"
        )
    return normalized


def _normalize_prior(prior: Sequence[float] | np.ndarray | None, *, n_classes: int) -> np.ndarray:
    if prior is None:
        return np.full(n_classes, 1.0 / float(n_classes), dtype=float)
    vector = np.asarray(prior, dtype=float).reshape(-1)
    if vector.shape[0] != n_classes:
        raise ValueError(f"source_prior must contain {n_classes} entries, got {vector.shape[0]}.")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("source_prior must contain finite non-negative values.")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("source_prior must have positive mass.")
    return vector / total


def _apply_bias(row: np.ndarray, *, bias: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(row, dtype=float), PROBABILITY_EPSILON, 1.0)) / float(temperature)
    logits = logits + np.asarray(bias, dtype=float)
    shifted = logits - np.max(logits)
    exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_logits / exp_logits.sum()


def _conditional_entropy_gradient(probability: np.ndarray) -> np.ndarray:
    safe = np.clip(np.asarray(probability, dtype=float), PROBABILITY_EPSILON, 1.0)
    entropy = -float(np.sum(safe * np.log(safe)))
    return -safe * (np.log(safe) + entropy)


def _update_running_marginal(running_marginal: np.ndarray, probability: np.ndarray, *, momentum: float) -> np.ndarray:
    updated = (1.0 - float(momentum)) * np.asarray(running_marginal, dtype=float) + float(momentum) * np.asarray(probability, dtype=float)
    return updated / updated.sum()


def _updated_bias(
    bias: np.ndarray,
    probability: np.ndarray,
    running_marginal: np.ndarray,
    prior: np.ndarray,
    *,
    learning_rate: float,
    entropy_weight: float,
    marginal_weight: float,
    bias_clip: float,
) -> np.ndarray:
    gradient = float(entropy_weight) * _conditional_entropy_gradient(probability)
    gradient = gradient + float(marginal_weight) * (np.asarray(running_marginal, dtype=float) - np.asarray(prior, dtype=float))
    gradient = gradient - float(np.mean(gradient))
    updated = np.asarray(bias, dtype=float) - float(learning_rate) * gradient
    updated = updated - float(np.mean(updated))
    if bias_clip > 0.0:
        updated = np.clip(updated, -float(bias_clip), float(bias_clip))
        updated = updated - float(np.mean(updated))
    return updated


def _normalize_positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed_float) or parsed_float % 1.0 != 0.0 or parsed_float < 1.0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed_float)


def _normalize_nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be non-negative and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed


def _normalize_positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _normalize_unit_interval_float(value: float | str, *, name: str, include_one: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        bracket = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must be finite in {bracket}.")
    parsed = float(value)
    upper_ok = parsed <= 1.0 if include_one else parsed < 1.0
    if not np.isfinite(parsed) or parsed < 0.0 or not upper_ok:
        bracket = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must be finite in {bracket}.")
    return parsed
