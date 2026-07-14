import pandas as pd

from neureptrace.semantic_stages import build_stage_report


def test_build_stage_report_escapes_markdown_table_cells():
    decoder = "linear|svm\nfold"
    emission_mode = "calibrated|raw"
    semantic_class = "animate|face\nstage"
    time_summary = pd.DataFrame(
        [
            {
                "decoder": decoder,
                "emission_mode": emission_mode,
                "true_class": semantic_class,
                "time": 0.12,
                "posterior_true_class_mean": 0.91,
                "viterbi_match_rate": 0.88,
                "n_sequences": 4,
            }
        ]
    )
    stages = pd.DataFrame(
        [
            {
                "decoder": decoder,
                "emission_mode": emission_mode,
                "semantic_class": semantic_class,
                "start_time": 0.10,
                "stop_time": 0.20,
                "duration": 0.10,
                "mean_posterior_true_class": 0.90,
                "mean_viterbi_match_rate": 0.87,
                "peak_time": 0.12,
                "n_subjects_min": 2,
                "n_sequences_min": 4,
            }
        ]
    )

    report = build_stage_report(
        time_summary,
        stages,
        posterior_threshold=0.7,
        match_threshold=0.7,
        min_duration=0.01,
    )

    assert "linear\\|svm fold" in report
    assert "calibrated\\|raw" in report
    assert "animate\\|face stage" in report
    assert decoder not in report
    assert semantic_class not in report
