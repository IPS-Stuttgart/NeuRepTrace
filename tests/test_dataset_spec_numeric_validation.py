from __future__ import annotations

import numpy as np
import pytest

from neureptrace.dataset_spec import dataset_spec_from_mapping


def _minimal_spec() -> dict[str, object]:
    return {
        "schema_version": "neureptrace.dataset.v1",
        "dataset_id": "toy",
        "subjects": [1],
        "splits": {
            "main": {
                "loader": "mne_epochs",
                "path_template": "sub-{subject}_epo.fif",
            }
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index_base", True),
        ("index_base", np.bool_(False)),
        ("chance_classes", False),
        ("chance_classes", 1.5),
        ("chance_classes", float("nan")),
    ],
)
def test_dataset_spec_rejects_invalid_label_numeric_fields(field: str, value: object) -> None:
    payload = _minimal_spec()
    payload["labels"] = {field: value}

    with pytest.raises(ValueError, match=rf"labels\.{field} must be"):
        dataset_spec_from_mapping(payload)


def test_dataset_spec_accepts_integral_label_numeric_strings() -> None:
    payload = _minimal_spec()
    payload["labels"] = {"index_base": "1", "chance_classes": "8"}

    spec = dataset_spec_from_mapping(payload)

    assert spec.labels.index_base == 1
    assert spec.labels.chance_classes == 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("false", False),
        ("0", False),
        ("no", False),
        (0, False),
        ("true", True),
        ("1", True),
        ("yes", True),
        (1, True),
    ],
)
def test_dataset_spec_parses_label_boolean_flag_without_truthy_string_bug(value: object, expected: bool) -> None:
    payload = _minimal_spec()
    payload["labels"] = {"subtract_one_when_no_null_class": value}

    spec = dataset_spec_from_mapping(payload)

    assert spec.labels.subtract_one_when_no_null_class is expected


@pytest.mark.parametrize("value", ["maybe", 2, -1, float("nan")])
def test_dataset_spec_rejects_invalid_label_boolean_flag(value: object) -> None:
    payload = _minimal_spec()
    payload["labels"] = {"subtract_one_when_no_null_class": value}

    with pytest.raises(ValueError, match=r"labels\.subtract_one_when_no_null_class must be a boolean"):
        dataset_spec_from_mapping(payload)


def test_dataset_spec_rejects_boolean_split_label_index_base() -> None:
    payload = _minimal_spec()
    splits = payload["splits"]
    assert isinstance(splits, dict)
    main_split = splits["main"]
    assert isinstance(main_split, dict)
    main_split["label_index_base"] = True

    with pytest.raises(ValueError, match="label_index_base must be an integer"):
        dataset_spec_from_mapping(payload)


@pytest.mark.parametrize(
    ("preprocessing", "message"),
    [
        ({"resample_hz": True}, "resample_hz must be a finite"),
        ({"window_size_s": float("inf")}, "window_size_s must be a finite"),
        ({"frequency_range_hz": [0.5, False]}, r"preprocessing_defaults\.frequency_range_hz\[1\] must be a finite"),
    ],
)
def test_dataset_spec_rejects_invalid_preprocessing_numeric_fields(preprocessing: dict[str, object], message: str) -> None:
    payload = _minimal_spec()
    payload["preprocessing_defaults"] = preprocessing

    with pytest.raises(ValueError, match=message):
        dataset_spec_from_mapping(payload)
