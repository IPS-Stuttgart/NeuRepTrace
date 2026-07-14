from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metadata import add_binary_label


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("source_column", 1),
        ("positive_pattern", True),
        ("negative_pattern", b"chair"),
        ("label_column", np.asarray(2)),
        ("positive_label", 3.5),
        ("negative_label", np.asarray(False)),
    ],
)
def test_add_binary_label_rejects_non_string_scalar_text_parameters(parameter: str, value: object) -> None:
    metadata = pd.DataFrame({"category": ["face", "chair"]})
    kwargs = {
        "source_column": "category",
        "positive_pattern": "face",
        "negative_pattern": "chair",
        "label_column": "condition",
        "positive_label": "target",
        "negative_label": "other",
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=rf"{parameter} must be a non-empty string"):
        add_binary_label(metadata, **kwargs)


def test_add_binary_label_accepts_zero_dimensional_numpy_string_scalars() -> None:
    metadata = pd.DataFrame({"category": ["face", "chair"]})

    labeled = add_binary_label(
        metadata,
        source_column=np.asarray("category"),
        positive_pattern=np.asarray("face"),
        negative_pattern=np.asarray("chair"),
        label_column=np.asarray("condition"),
        positive_label=np.asarray("target"),
        negative_label=np.asarray("other"),
    )

    assert labeled["condition"].tolist() == ["target", "other"]
