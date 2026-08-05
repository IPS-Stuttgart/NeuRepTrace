"""Source-only domain-risk summary helpers."""

from __future__ import annotations

import numpy as np

from neureptrace.decoding._domain_labels import _as_domain_vector, _domain_mask, _unique_domain_labels

DOMAIN_INVARIANCE_PROTOCOL = "source_only_domain_risk_summary"


def domain_risk_summary(losses, domains):
    """Return per-domain mean loss, overall mean, and variance."""
    loss_values = np.asarray(losses, dtype=float).reshape(-1)
    try:
        domain_values = _as_domain_vector(domains, expected_length=loss_values.shape[0])
    except ValueError as exc:
        raise ValueError("losses and domains must contain the same rows") from exc
    if loss_values.size == 0 or not np.all(np.isfinite(loss_values)):
        raise ValueError("losses must contain finite values")
    levels = _unique_domain_labels(domain_values)
    if len(levels) < 2:
        raise ValueError("at least two source domains are required")
    per_domain = {}
    for level in levels:
        per_domain[level] = _stable_mean(loss_values[_domain_mask(domain_values, level)])
    values = np.asarray(tuple(per_domain.values()), dtype=float)
    return {
        "domain_risks": per_domain,
        "mean_risk": _stable_mean(values),
        "risk_variance": _stable_variance(values),
        "protocol": DOMAIN_INVARIANCE_PROTOCOL,
        "uses_target_features": False,
        "uses_target_labels": False,
    }


def _stable_mean(values: np.ndarray) -> float:
    """Return the arithmetic mean without overflowing finite same-sign inputs."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    normalized = values / scale
    normalized_mean = float(np.mean(normalized))
    lower = float(np.min(normalized))
    upper = float(np.max(normalized))
    return float(np.clip(normalized_mean, lower, upper) * scale)


def _stable_variance(values: np.ndarray) -> float:
    """Return the population variance when it is representable in ``float64``."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    normalized_variance = float(np.var(values / scale))
    if normalized_variance == 0.0:
        return 0.0
    standard_deviation = float(np.sqrt(normalized_variance) * scale)
    if standard_deviation > np.sqrt(np.finfo(float).max):
        return float("inf")
    return float(standard_deviation * standard_deviation)
