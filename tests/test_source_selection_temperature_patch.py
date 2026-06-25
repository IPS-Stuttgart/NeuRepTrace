import numpy as np
import pytest

from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity


_SOURCE_FEATURES = np.asarray(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [4.0, 4.0],
        [4.2, 4.1],
    ],
    dtype=float,
)
_SOURCE_DOMAINS = np.asarray(["near", "near", "far", "far"], dtype=object)
_TARGET_FEATURES = np.asarray(
    [
        [0.05, 0.1],
        [0.2, 0.0],
    ],
    dtype=float,
)


@pytest.mark.parametrize("temperature", [True, False, np.bool_(True), np.bool_(False)])
def test_source_selection_rejects_boolean_softmax_temperature(temperature):
    with pytest.raises(ValueError, match="softmax_temperature.*boolean"):
        select_source_domains_by_target_similarity(
            _SOURCE_FEATURES,
            _SOURCE_DOMAINS,
            _TARGET_FEATURES,
            softmax_temperature=temperature,
        )


def test_source_selection_temperature_patch_preserves_valid_temperatures():
    result = select_source_domains_by_target_similarity(
        _SOURCE_FEATURES,
        _SOURCE_DOMAINS,
        _TARGET_FEATURES,
        softmax_temperature="0.5",
    )

    assert result.selected_domains == ("near", "far")
    assert np.all(np.isfinite(result.sample_weights))
    assert np.all(result.sample_weights > 0.0)
