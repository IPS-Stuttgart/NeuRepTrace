from __future__ import annotations

from neureptrace.openneuro_decode_diagnostics import _optional_unique_bool, _provenance_value


def test_manifest_provenance_list_values_are_normalized():
    manifest = {
        "alignment_valid_for_benchmark": [True, True],
        "ensemble_weights": [0.5, 0.3, 0.2],
    }

    assert _provenance_value(manifest, {}, "ensemble_weights") == "0.5|0.3|0.2"
    assert _optional_unique_bool(
        _provenance_value(manifest, {}, "alignment_valid_for_benchmark"),
        column="alignment_valid_for_benchmark",
    ) is True
