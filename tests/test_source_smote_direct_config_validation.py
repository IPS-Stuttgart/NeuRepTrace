from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import SourceSmoteConfig


def test_source_smote_config_normalizes_direct_dataclass_values() -> None:
    cfg = SourceSmoteConfig(
        synthetic_per_class="2",  # type: ignore[arg-type]
        cross_domain_partner="false",  # type: ignore[arg-type]
        preserve_original="0",  # type: ignore[arg-type]
        random_state=np.asarray("none"),
        jitter_std="0.25",  # type: ignore[arg-type]
    )

    assert cfg.synthetic_per_class == 2
    assert isinstance(cfg.synthetic_per_class, int)
    assert cfg.cross_domain_partner is False
    assert cfg.preserve_original is False
    assert cfg.random_state is None
    assert cfg.jitter_std == pytest.approx(0.25)
    assert cfg.enabled is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"synthetic_per_class": True}, "synthetic_per_class must be an integer"),
        ({"cross_domain_partner": np.asarray([True])}, "cross_domain_partner must be a boolean"),
        ({"random_state": np.asarray([0])}, "random_state must be a non-negative integer or none"),
        ({"jitter_std": -0.1}, "jitter_std must be non-negative and finite"),
    ],
)
def test_source_smote_config_rejects_invalid_direct_dataclass_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SourceSmoteConfig(**kwargs)  # type: ignore[arg-type]
