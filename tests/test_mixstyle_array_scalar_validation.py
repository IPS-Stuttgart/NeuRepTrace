from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mixstyle import augment_source_mixstyle, source_mixstyle_config as feature_mixstyle_config
from neureptrace.decoding.source_mixstyle import SourceMixStyleConfig, augment_source_domains_mixstyle, source_mixstyle_config as domain_mixstyle_config


def _feature_data():
    return (
        np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float),
        np.asarray(["a", "b", "a", "b"], dtype=object),
        np.asarray(["s1", "s1", "s2", "s2"], dtype=object),
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("augmentations_per_row", {"augmentations_per_row": np.asarray([1])}),
        ("alpha", {"alpha": np.asarray([0.2])}),
        ("random_state", {"random_state": np.asarray(13)}),
    ],
)
def test_feature_mixstyle_rejects_array_valued_numeric_controls(name: str, kwargs: dict[str, object]) -> None:
    features, labels, domains = _feature_data()

    with pytest.raises(ValueError, match=rf"{name}.*scalar"):
        feature_mixstyle_config(**kwargs)

    with pytest.raises(ValueError, match=rf"{name}.*scalar"):
        augment_source_mixstyle(features, labels, domains, **kwargs)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("mixes_per_row", {"mixes_per_row": np.asarray([1])}),
        ("alpha", {"alpha": np.asarray([0.3])}),
        ("style_strength", {"style_strength": np.asarray([0.5])}),
        ("synthetic_weight", {"synthetic_weight": np.asarray([1.0])}),
        ("random_state", {"random_state": np.asarray(13)}),
    ],
)
def test_domain_mixstyle_rejects_array_valued_numeric_config(name: str, config: dict[str, object]) -> None:
    features, labels, domains = _feature_data()

    with pytest.raises(ValueError, match=rf"{name}.*scalar"):
        domain_mixstyle_config(**config)

    with pytest.raises(ValueError, match=rf"{name}.*scalar"):
        augment_source_domains_mixstyle(features, labels, domains, config=config)


def test_domain_mixstyle_rejects_array_valued_dataclass_config() -> None:
    features, labels, domains = _feature_data()
    config = SourceMixStyleConfig(mixes_per_row=np.asarray([1]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"mixes_per_row.*scalar"):
        augment_source_domains_mixstyle(features, labels, domains, config=config)
