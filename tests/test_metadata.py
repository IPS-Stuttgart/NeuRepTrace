import pandas as pd
import pytest

from neureptrace.metadata import add_binary_label


def test_add_binary_label_marks_positive_and_default_negative():
    metadata = pd.DataFrame({"category": ["face/person", "chair", "car"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="face|person",
        label_column="is_face",
        positive_label="face",
        negative_label="object",
    )

    assert labeled["is_face"].tolist() == ["face", "object", "object"]


def test_add_binary_label_with_negative_pattern_leaves_unmatched_missing():
    metadata = pd.DataFrame({"category": ["face", "chair", "unknown"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="face",
        negative_pattern="chair",
        label_column="condition",
    )

    assert labeled["condition"].tolist()[:2] == ["positive", "negative"]
    assert pd.isna(labeled["condition"].tolist()[2])


def test_add_binary_label_prefers_positive_when_patterns_overlap():
    metadata = pd.DataFrame({"category": ["face/person", "chair", "person-chair", "unknown"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="face|person",
        negative_pattern="chair|person",
        label_column="condition",
        positive_label="face",
        negative_label="object",
    )

    assert labeled["condition"].tolist()[:3] == ["face", "object", "face"]
    assert pd.isna(labeled["condition"].tolist()[3])


def test_add_binary_label_respects_case_sensitive_matching():
    metadata = pd.DataFrame({"category": ["Face", "face"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="Face",
        label_column="condition",
        case_sensitive=True,
    )

    assert labeled["condition"].tolist() == ["positive", "negative"]


def test_add_binary_label_rejects_existing_label_column():
    metadata = pd.DataFrame({"category": ["face"], "condition": ["old"]})

    with pytest.raises(ValueError, match="already exists"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="face",
            label_column="condition",
        )


def test_add_binary_label_rejects_empty_patterns():
    metadata = pd.DataFrame({"category": ["face", "chair"]})

    with pytest.raises(ValueError, match="positive_pattern must be a non-empty string"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="",
            label_column="condition",
        )

    with pytest.raises(ValueError, match="negative_pattern must be a non-empty string"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="face",
            negative_pattern="   ",
            label_column="condition",
        )


def test_add_binary_label_rejects_invalid_regex():
    metadata = pd.DataFrame({"category": ["face", "chair"]})

    with pytest.raises(ValueError, match="positive_pattern must be a valid regular expression"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="[",
            label_column="condition",
        )


def test_add_binary_label_rejects_ambiguous_label_values():
    metadata = pd.DataFrame({"category": ["face", "chair"]})

    with pytest.raises(ValueError, match="positive_label must be a non-empty string"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="face",
            label_column="condition",
            positive_label=" ",
        )

    with pytest.raises(ValueError, match="positive_label and negative_label must be distinct"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="face",
            label_column="condition",
            positive_label="target",
            negative_label="target",
        )
