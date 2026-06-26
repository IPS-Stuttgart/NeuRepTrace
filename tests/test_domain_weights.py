from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.domain_importance import DOMAIN_IMPORTANCE_CATEGORY, domain_importance_config, fit_domain_classifier_importance_weights


def test_domain_weights_are_category2_and_normalized() -> None:
    source = np.asarray([[-4.0], [-3.8], [0.0], [0.2], [4.0], [4.2]], dtype=float)
    target = np.asarray([[3.7], [4.1], [4.4]], dtype=float)

    result = fit_domain_classifier_importance_weights(source, target, config={"clip": "0.01,100", "normalize": True})

    assert result.sample_weights.shape == (6,)
    assert np.all(result.sample_weights >= 0.0)
    assert np.isclose(np.mean(result.sample_weights), 1.0)
    assert result.sample_weights[-1] > result.sample_weights[0]
    assert result.metadata["domain_importance_protocol_category"] == DOMAIN_IMPORTANCE_CATEGORY
    assert result.metadata["domain_importance_uses_target_features"] is True
    assert result.metadata["domain_importance_uses_target_labels"] is False


def test_domain_weights_record_clip_none() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    target = np.asarray([[0.5], [2.5]], dtype=float)

    result = fit_domain_classifier_importance_weights(source, target, config={"clip": None})

    assert result.metadata["domain_importance_clip_min"] == ""
    assert result.metadata["domain_importance_clip_max"] == ""
    assert np.isclose(result.sample_weights.mean(), 1.0)


def test_domain_importance_config_parses_boolean_strings() -> None:
    cfg = domain_importance_config(normalize="false", account_for_sample_priors="off")

    assert cfg.normalize is False
    assert cfg.account_for_sample_priors is False


def test_domain_importance_config_rejects_ambiguous_boolean_strings() -> None:
    with pytest.raises(ValueError, match="normalize"):
        domain_importance_config(normalize="sometimes")
