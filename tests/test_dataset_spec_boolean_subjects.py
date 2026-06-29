from __future__ import annotations

import numpy as np
import pytest

from neureptrace.dataset_spec import dataset_spec_from_mapping, parse_subjects


@pytest.mark.parametrize(
    "subjects",
    [
        True,
        [False],
        [np.bool_(True)],
        {"include": [1, True]},
        {"include": "1-2", "exclude": [np.bool_(False)]},
    ],
)
def test_parse_subjects_rejects_boolean_identifiers(subjects: object) -> None:
    with pytest.raises(ValueError, match="boolean identifiers"):
        parse_subjects(subjects)


def test_dataset_spec_from_mapping_rejects_boolean_subject_identifier() -> None:
    payload = {
        "schema_version": "neureptrace.dataset.v1",
        "dataset_id": "toy",
        "subjects": [True],
        "splits": {
            "main": {
                "loader": "csv_feature_matrix",
                "path_template": "sub-{subject}.csv",
            }
        },
    }

    with pytest.raises(ValueError, match="boolean identifiers"):
        dataset_spec_from_mapping(payload)
