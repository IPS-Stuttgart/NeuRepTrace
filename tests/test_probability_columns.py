import pandas as pd
import pytest

from neureptrace.observations import probability_columns


def test_probability_columns_rejects_duplicate_numeric_class_labels() -> None:
    frame = pd.DataFrame({"prob_class_01": [0.4], "prob_class_1": [0.6]})

    with pytest.raises(ValueError, match="unique class labels"):
        probability_columns(frame)


def test_probability_columns_allows_non_numeric_suffixes_without_relabeling() -> None:
    frame = pd.DataFrame({"prob_class_cat": [0.4], "prob_class_1": [0.6]})

    assert probability_columns(frame) == ("prob_class_1", "prob_class_cat")
