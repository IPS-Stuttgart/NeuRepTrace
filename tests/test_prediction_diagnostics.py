import pandas as pd

from neureptrace.results import augment_prediction_ranks, build_prediction_diagnostics, ranked_label_metrics


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2"],
            "decoder": ["logistic"] * 4,
            "true_label": [0, 1, 2, 1],
            "predicted_label": [0, 2, 2, 0],
            "prob_class_0": [0.80, 0.10, 0.10, 0.45],
            "prob_class_1": [0.15, 0.40, 0.20, 0.40],
            "prob_class_2": [0.05, 0.50, 0.70, 0.15],
        }
    )


def test_augment_prediction_ranks_adds_row_level_rank_fields():
    ranked = augment_prediction_ranks(_prediction_frame())

    assert ranked["rank1_label"].tolist() == [0, 2, 2, 0]
    assert ranked["true_label_rank"].tolist() == [1.0, 2.0, 1.0, 2.0]
    assert ranked["top1_hit"].tolist() == [True, False, True, False]
    assert ranked["top2_hit"].tolist() == [True, True, True, True]


def test_ranked_label_metrics_summarizes_top_k_by_group():
    summary = ranked_label_metrics(_prediction_frame(), group_columns=("decoder",))

    row = summary.iloc[0]
    assert row["decoder"] == "logistic"
    assert row["n_rows"] == 4
    assert round(float(row["top1_accuracy"]), 3) == 0.5
    assert round(float(row["top2_accuracy"]), 3) == 1.0
    assert round(float(row["mean_true_label_rank"]), 3) == 1.5


def test_build_prediction_diagnostics_returns_confusion_recall_and_rank_tables():
    diagnostics = build_prediction_diagnostics(
        _prediction_frame(),
        group_columns=("decoder",),
        participant_column="participant",
    )

    assert set(diagnostics) == {
        "predictions",
        "confusion",
        "per_class_recall",
        "confusion_pairs",
        "rank_summary",
        "category_enrichment",
        "category_matrix",
    }

    confusion = diagnostics["confusion"]
    one_as_two = confusion[(confusion["true_label"] == 1) & (confusion["predicted_label"] == 2)].iloc[0]
    assert one_as_two["count"] == 1

    recall = diagnostics["per_class_recall"]
    label_one = recall[recall["true_label"] == 1].iloc[0]
    assert label_one["support"] == 2
    assert label_one["n_correct"] == 0
    assert label_one["recall"] == 0.0
    assert label_one["n_participants"] == 2

    rank_summary = diagnostics["rank_summary"].iloc[0]
    assert rank_summary["top2_accuracy"] == 1.0
    assert diagnostics["category_enrichment"].empty
    assert diagnostics["category_matrix"].empty


def test_build_prediction_diagnostics_adds_metadata_conditioned_tables():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2", "p3"],
            "decoder": ["svm"] * 5,
            "true_stimulus": [1, 1, 2, 3, 4],
            "predicted_stimulus": [2, 2, 1, 4, 3],
        }
    )
    metadata = pd.DataFrame(
        {
            "stimulus": [1, 2, 3, 4],
            "name": ["cat", "dog", "cup", "bottle"],
            "semantic_category": ["animal", "animal", "object", "object"],
        }
    )

    diagnostics = build_prediction_diagnostics(
        predictions,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=("decoder",),
        participant_column="participant",
        metadata_frame=metadata,
        category_columns=("semantic_category",),
        label_prefix="stimulus",
        n_permutations=0,
    )

    pair = diagnostics["confusion_pairs"]
    stimulus_pair = pair[(pair["stimulus_a"] == 1) & (pair["stimulus_b"] == 2)].iloc[0]
    assert stimulus_pair["total_confusions"] == 3
    assert bool(stimulus_pair["same_semantic_category"]) is True

    enrichment = diagnostics["category_enrichment"].iloc[0]
    assert enrichment["category_column"] == "semantic_category"
    assert enrichment["same_category_errors"] == 5

    matrix = diagnostics["category_matrix"]
    animal = matrix[(matrix["true_category"] == "animal") & (matrix["predicted_category"] == "animal")].iloc[0]
    assert animal["count"] == 3
