import numpy as np
import pandas as pd
import pytest

from neureptrace.metadata import add_binary_label


@pytest.mark.parametrize(
    ("case_sensitive", "expected"),
    [
        (np.bool_(True), ["positive", "negative"]),
        (np.asarray(False), ["positive", "positive"]),
    ],
)
def test_add_binary_label_accepts_numpy_boolean_scalars(case_sensitive, expected):
    metadata = pd.DataFrame({"category": ["Face", "face"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="Face",
        label_column="condition",
        case_sensitive=case_sensitive,
    )

    assert labeled["condition"].tolist() == expected


@pytest.mark.parametrize(
    "case_sensitive",
    [
        "false",
        0,
        1,
        None,
        [False],
        np.asarray([True]),
    ],
)
def test_add_binary_label_rejects_non_boolean_case_sensitive_values(case_sensitive):
    metadata = pd.DataFrame({"category": ["Face", "face"]})

    with pytest.raises(ValueError, match="case_sensitive must be a boolean"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="Face",
            label_column="condition",
            case_sensitive=case_sensitive,
        )
