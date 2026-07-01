from __future__ import annotations

from neureptrace.decoding.source_polynomial import SourcePolynomialConfig, fit_source_polynomial_reference


def test_polynomial_revalidates_direct_dataclass_config() -> None:
    disabled = str(False).lower()
    enabled = str(True).lower()
    cfg = SourcePolynomialConfig(
        include_bias=disabled,
        include_original=disabled,
        include_squares=disabled,
        include_interactions=enabled,
        max_interactions=1,
    )

    reference = fit_source_polynomial_reference(2, config=cfg)

    assert reference.config.include_bias is False
    assert reference.config.include_original is False
    assert reference.config.include_squares is False
    assert reference.config.include_interactions is True
    assert reference.output_names == ("x0*x1",)
