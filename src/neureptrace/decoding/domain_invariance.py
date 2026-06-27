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
        per_domain[level] = float(np.mean(loss_values[_domain_mask(domain_values, level)]))
    values = np.asarray(tuple(per_domain.values()), dtype=float)
    return {
        "domain_risks": per_domain,
        "mean_risk": float(np.mean(values)),
        "risk_variance": float(np.var(values)),
        "protocol": DOMAIN_INVARIANCE_PROTOCOL,
        "uses_target_features": False,
        "uses_target_labels": False,
    }
