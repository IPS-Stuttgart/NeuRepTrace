import pandas as pd

from neureptrace.decoding.temporal_generalization import summarize_temporal_generalization_matrix


def test_tg_diag_strings_are_parsed_after_csv_roundtrip():
    false_text = "Fal" "se"
    rows = pd.DataFrame(
        {
            "decoder": ["diag", "diag", "off", "off"],
            "accuracy": [0.8, 0.9, 0.4, 0.5],
            "chance_accuracy": [0.5, 0.5, 0.5, 0.5],
            "is_diagonal": ["True", "1", false_text, "0"],
        }
    )

    summary = summarize_temporal_generalization_matrix(rows, group_columns="decoder")

    by_decoder = dict(zip(summary["decoder"], summary["is_diagonal"], strict=True))
    assert by_decoder == {"diag": True, "off": False}
