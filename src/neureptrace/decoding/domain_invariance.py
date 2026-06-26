"""Source-only domain-risk summary helpers."""

from __future__ import annotations

import numpy as np

DOMAIN_INVARIANCE_PROTOCOL = "source_only_domain_risk_summary"


def domain_risk_summary(losses, domains):
    """Return per-domain mean loss, overall mean, and variance."""
    loss_values = np.asarray(losses, dtype=float).reshape(-1)
    domain_values = np.asarray(domains, dtype=object).reshape(-1)
    if loss_values.shape[0] != domain_values.shape[0]:
        raise ValueError("losses and domains must contain the same rows")
    if loss_values.size == 0 or not np.all(np.isfinite(loss_values)):
        raise ValueError("losses must contain finite values")
    levels = tuple(dict.fromkeys(domain_values.tolist()))
    if len(levels) < 2:
        raise ValueError("at least two source domains are required")
    per_domain = {level: float(np.mean(loss_values[domain_values == level])) for level in levels}
    values = np.asarray(tuple(per_domain.values()), dtype=float)
    return {
        "domain_risks": per_domain,
        "mean_risk": float(np.mean(values)),
        "risk_variance": float(np.var(values)),
        "protocol": DOMAIN_INVARIANCE_PROTOCOL,
        "uses_target_features": False,
        "uses_target_labels": False,
    }
