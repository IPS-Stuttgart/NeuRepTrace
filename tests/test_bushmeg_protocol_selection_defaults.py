from __future__ import annotations

from neureptrace.bushmeg_all_protocols import _configured_method_names, _selected_methods


def test_default_selection_respects_explicit_protocol3_request() -> None:
    selected = _selected_methods(
        all_protocols={},
        methods=None,
        protocols="3",
        include_oracle=False,
    )

    assert selected
    assert all(spec.protocol_category == 3 for spec in selected)
    selected_names = {spec.method for spec in selected}
    assert "few_shot_target_calibrated_decoder_k1" in selected_names
    assert "source_plus_target_calibration_logistic_k1" in selected_names

    configured_names = _configured_method_names(
        all_protocols={},
        methods=None,
        protocols="3",
        include_oracle=False,
    )
    assert configured_names == selected_names


def test_default_selection_without_explicit_protocols_keeps_existing_protocol1_2_defaults() -> None:
    selected = _selected_methods(
        all_protocols={},
        methods=None,
        protocols=None,
        include_oracle=False,
    )

    assert selected
    categories = {spec.protocol_category for spec in selected}
    assert categories.issubset({1, 2})
    assert 3 not in categories
