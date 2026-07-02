from __future__ import annotations

import pytest

from neureptrace.decoding.source_mahalanobis import SourceMahalanobisConfig


def test_direct_source_mahalanobis_config_normalizes_values() -> None:
    cfg = SourceMahalanobisConfig(regularization="0.01", prior="flat", temperature="2.0")  # type: ignore[arg-type]

    assert cfg.regularization == pytest.approx(0.01)
    assert cfg.prior == "uniform"
    assert cfg.temperature == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"regularization": -1.0}, "regularization"),
        ({"prior": "posterior"}, "prior"),
        ({"temperature": float("nan")}, "temperature"),
    ],
)
def test_direct_source_mahalanobis_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SourceMahalanobisConfig(**kwargs)  # type: ignore[arg-type]
