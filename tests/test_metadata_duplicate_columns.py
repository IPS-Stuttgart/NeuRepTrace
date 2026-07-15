import pandas as pd
import pytest

from neureptrace.metadata import add_binary_label


def test_add_binary_label_rejects_duplicate_source_columns():
    metadata = pd.DataFrame([["face", "chair"]], columns=["category", "category"])

    with pytest.raises(ValueError, match="must identify exactly one"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="face",
            label_column="condition",
        )
