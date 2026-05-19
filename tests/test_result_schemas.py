import pandas as pd
import pytest

from neureptrace.results import (
    OUTER_FOLD_SCORE_SCHEMA,
    PROBABILITY_OBSERVATION_SCHEMA,
    RESULT_TABLE_SCHEMAS,
    canonicalize_result_table,
    schema_as_frame,
    validate_result_table,
    write_result_table,
)


def _outer_fold_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "custom_project_column": ["kept"],
            "accuracy": [0.80],
            "time": [0.15],
            "subject": ["sub-01"],
            "log_loss": [0.42],
            "brier": [0.18],
            "ece": [0.04],
            "n_test": [25],
        }
    )


def test_validate_result_table_accepts_outer_fold_scores():
    frame = _outer_fold_scores()

    returned = validate_result_table(frame, OUTER_FOLD_SCORE_SCHEMA)

    assert returned is frame


def test_validate_result_table_reports_missing_required_columns():
    with pytest.raises(ValueError, match="outer_fold_scores.*missing required columns"):
        validate_result_table(pd.DataFrame({"subject": ["sub-01"]}), OUTER_FOLD_SCORE_SCHEMA)


def test_validate_result_table_checks_integer_columns():
    frame = _outer_fold_scores()
    frame["n_test"] = [2.5]

    with pytest.raises(ValueError, match="n_test.*integer values"):
        validate_result_table(frame, OUTER_FOLD_SCORE_SCHEMA)


def test_probability_observations_require_probability_columns():
    frame = pd.DataFrame({"time": [0.0], "true_label": [0]})

    with pytest.raises(ValueError, match="prob_class_"):
        validate_result_table(frame, PROBABILITY_OBSERVATION_SCHEMA)


def test_probability_observations_validate_probability_columns():
    frame = pd.DataFrame({"time": [0.0], "true_label": [0], "prob_class_0": [0.8], "prob_class_1": [0.2]})

    validate_result_table(frame, PROBABILITY_OBSERVATION_SCHEMA)


def test_canonicalize_result_table_orders_known_columns_before_project_columns():
    frame = _outer_fold_scores()

    canonical = canonicalize_result_table(frame, OUTER_FOLD_SCORE_SCHEMA)

    assert canonical.columns[:6].tolist() == ["subject", "time", "accuracy", "log_loss", "brier", "ece"]
    assert canonical.columns[-1] == "custom_project_column"


def test_write_result_table_validates_and_writes_canonical_csv(tmp_path):
    frame = _outer_fold_scores()
    out_path = tmp_path / "outer_fold_scores.csv"

    canonical = write_result_table(frame, out_path, OUTER_FOLD_SCORE_SCHEMA)
    loaded = pd.read_csv(out_path)

    assert canonical.columns.tolist() == loaded.columns.tolist()
    assert loaded.loc[0, "subject"] == "sub-01"


def test_schema_as_frame_documents_registered_schemas():
    assert "provenance" in RESULT_TABLE_SCHEMAS

    documented = schema_as_frame(OUTER_FOLD_SCORE_SCHEMA)

    assert {"table", "column", "kind", "required", "nullable", "description"} <= set(documented.columns)
    assert documented.loc[documented["column"] == "subject", "required"].iloc[0]
