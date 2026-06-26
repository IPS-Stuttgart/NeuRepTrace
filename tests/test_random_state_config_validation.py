from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mixstyle import augment_source_mixstyle, source_mixstyle_config
from neureptrace.decoding.source_mixstyle import (
    SourceMixStyleConfig,
    augment_source_domains_mixstyle,
    source_mixstyle_config as source_domain_mixstyle_config,
)


def _toy_sources():
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [10.0, 10.0],
            [11.0, 10.5],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "b", "a", "b"], dtype=object)
    domains = np.asarray(["s1", "s1", "s2", "s2"], dtype=object)
    return features, labels, domains


@pytest.mark.parametrize("bad_random_state", [[1, 2], np.asarray([1, 2], dtype=int)])
def test_feature_mixstyle_rejects_nonscalar_random_state(bad_random_state) -> None:
    features, labels, domains = _toy_sources()

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        source_mixstyle_config(random_state=bad_random_state)

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        augment_source_mixstyle(features, labels, domains, random_state=bad_random_state)


@pytest.mark.parametrize("bad_random_state", [[1, 2], np.asarray([1, 2], dtype=int)])
def test_source_domain_mixstyle_rejects_nonscalar_random_state(bad_random_state) -> None:
    features, labels, domains = _toy_sources()

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        source_domain_mixstyle_config(random_state=bad_random_state)

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        augment_source_domains_mixstyle(
            features,
            labels,
            domains,
            config={"random_state": bad_random_state},
        )

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        augment_source_domains_mixstyle(
            features,
            labels,
            domains,
            config=SourceMixStyleConfig(random_state=bad_random_state),  # type: ignore[arg-type]
        )


def test_random_state_none_sentinel_still_disables_seeding() -> None:
    features, labels, domains = _toy_sources()

    assert source_mixstyle_config(random_state="none").random_state is None
    assert source_domain_mixstyle_config(random_state="none").random_state is None

    feature_result = augment_source_mixstyle(features, labels, domains, random_state="none")
    domain_result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"random_state": "none"},
    )

    assert feature_result.features.shape[0] == 8
    assert domain_result.features.shape[0] == 8
