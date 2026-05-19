from pathlib import Path

import pandas as pd

from neureptrace.results import diagnostic_summary_tables, prediction_diagnostic_table, write_diagnostic_exports


def _prediction_frame() -> pd.DataFrame:
    return prediction_diagnostic_table(
        ["dog", "cat", "dog"],
        ["dog", "cat", "cup"],
        scores=[
            [0.1, 0.9, 0.3],
            [0.4, 0.2, 0.1],
            [0.1, 0.2, 0.7],
        ],
        classes=["cat", "dog", "cup"],
        sample_ids=["trial-1", "trial-2", "trial-3"],
        group_values={"participant": ["p1", "p1", "p2"], "window": 0.15},
        top_k=(1, 2),
        row_top_k=2,
        class_column="stimulus",
    )


def test_prediction_diagnostic_table_adds_scores_and_true_label_ranks():
    predictions = _prediction_frame()

    assert predictions["sample_index"].tolist() == ["trial-1", "trial-2", "trial-3"]
    assert predictions["correct"].tolist() == [True, True, False]
    assert predictions["true_label_rank"].tolist() == [1.0, 1.0, 2.0]
    assert predictions["top_1_hit"].tolist() == [True, True, False]
    assert predictions["top_2_hit"].tolist() == [True, True, True]
    assert predictions["rank1_stimulus"].tolist() == ["dog", "cat", "cup"]
    assert predictions["score_class_dog"].tolist() == [0.9, 0.2, 0.2]


def test_diagnostic_summary_tables_build_confusion_per_class_and_rank_summaries():
    predictions = _prediction_frame()

    tables = diagnostic_summary_tables(predictions, group_columns=("window",), participant_column="participant", top_k=(1, 2))

    dog_as_cup = tables.confusion[(tables.confusion["true_label"] == "dog") & (tables.confusion["predicted_label"] == "cup")].iloc[0]
    assert dog_as_cup["count"] == 1

    dog = tables.per_class.loc[tables.per_class["true_label"] == "dog"].iloc[0]
    assert dog["n_trials"] == 2
    assert dog["n_correct"] == 1
    assert round(float(dog["accuracy"]), 3) == 0.5
    assert dog["n_participants"] == 2

    rank = tables.rank_summary.iloc[0]
    assert rank["n_rows"] == 3
    assert rank["n_ranked"] == 3
    assert round(float(rank["mean_true_label_rank"]), 3) == 1.333
    assert round(float(rank["top_1_accuracy"]), 3) == 0.667
    assert round(float(rank["top_2_accuracy"]), 3) == 1.0


def test_write_diagnostic_exports_writes_stable_csv_set(tmp_path: Path):
    predictions = _prediction_frame()

    tables = write_diagnostic_exports(
        predictions,
        tmp_path,
        group_columns="window",
        participant_column="participant",
        top_k=(1, 2),
    )

    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "confusion.csv").exists()
    assert (tmp_path / "per_class.csv").exists()
    assert (tmp_path / "rank_summary.csv").exists()
    assert (tmp_path / "confusion_pairs.csv").exists()
    assert pd.read_csv(tmp_path / "predictions.csv")["true_label"].tolist() == predictions["true_label"].tolist()
    assert pd.read_csv(tmp_path / "rank_summary.csv")["top_2_accuracy"].tolist() == tables.rank_summary["top_2_accuracy"].tolist()
