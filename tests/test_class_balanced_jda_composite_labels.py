import numpy as np
import pytest

from neureptrace.decoding.class_balanced_jda import class_balanced_source_indices, fit_class_balanced_jda


_SOURCE_FEATURES = np.asarray(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.1],
        [4.0, 4.0],
        [4.1, 4.2],
    ],
    dtype=float,
)
_SOURCE_LABELS = np.asarray(
    [
        ["left", "hand"],
        ["left", "hand"],
        ["right", "hand"],
        ["right", "hand"],
        ["right", "hand"],
    ],
    dtype=object,
)
_TARGET_FEATURES = np.asarray(
    [
        [0.05, 0.1],
        [4.05, 4.1],
    ],
    dtype=float,
)


def test_class_balanced_source_indices_accepts_array_composite_labels():
    result = class_balanced_source_indices(_SOURCE_LABELS, strategy="oversample", random_state=0)

    assert result.classes == (("left", "hand"), ("right", "hand"))
    assert result.original_counts == (2, 3)
    assert result.balanced_counts == (3, 3)
    assert result.indices.shape == (6,)


def test_class_balanced_source_indices_accepts_string_random_state():
    numeric_result = class_balanced_source_indices(_SOURCE_LABELS, strategy="oversample", random_state=0)
    string_result = class_balanced_source_indices(_SOURCE_LABELS, strategy="oversample", random_state="0")

    assert string_result.indices.tolist() == numeric_result.indices.tolist()
    assert string_result.random_state == 0


@pytest.mark.parametrize("bad_random_state", [True, False, -1, 1.5, "1.5", "nan", "seed"])
def test_class_balanced_source_indices_rejects_invalid_random_state(bad_random_state):
    with pytest.raises(ValueError, match="random_state"):
        class_balanced_source_indices(_SOURCE_LABELS, strategy="oversample", random_state=bad_random_state)


def test_fit_class_balanced_jda_accepts_array_composite_labels():
    result = fit_class_balanced_jda(
        _SOURCE_FEATURES,
        _SOURCE_LABELS,
        _TARGET_FEATURES,
        balance_strategy="oversample",
        balance_random_state="0",
        n_components=1,
        max_iterations=2,
    )

    assert result.classes == (("left", "hand"), ("right", "hand"))
    assert result.source_features.shape == (6, 1)
    assert result.target_features.shape == (2, 1)
    assert result.metadata["class_balanced_jda"] is True
    assert result.metadata["class_balanced_jda_random_state"] == 0
    assert result.metadata["class_balanced_jda_uses_target_labels"] is False
    assert result.metadata["class_balanced_jda_balanced_counts"] == "3|3"
