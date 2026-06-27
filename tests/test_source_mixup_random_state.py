from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.decoding import source_mixup


def test_source_mixup_config_rejects_negative_random_state() -> None:
    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        source_mixup.source_mixup_config(random_state=-1)


def test_source_mixup_config_accepts_none_random_state_strings() -> None:
    assert source_mixup.source_mixup_config(random_state="none").random_state is None
    assert source_mixup.source_mixup_config(random_state="null").random_state is None


def test_source_mixup_dataclass_config_rejects_negative_random_state_before_rng() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["class_a", "class_a", "class_b", "class_b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)
    cfg = source_mixup.SourceMixUpConfig(synthetic_per_class=1, random_state=-1)

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        source_mixup.augment_source_with_mixup(
            features,
            labels,
            source_domains=domains,
            config=cfg,
        )
