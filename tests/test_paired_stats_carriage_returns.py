import pandas as pd

from neureptrace.paired_stats import build_paired_stats_report


def test_paired_stats_report_collapses_carriage_returns() -> None:
    statistics = pd.DataFrame(
        [
            {
                "emission_mode": "calibrated\rlegacy",
                "decoder_a": "lda\rlegacy",
                "decoder_b": "logistic\r\nmodern",
                "metric": "effect_accuracy",
                "preferred_direction": "higher",
                "n_subjects": 3,
                "decoder_a_mean": 0.5,
                "decoder_b_mean": 0.6,
                "mean_difference_a_minus_b": -0.1,
                "sign_flip_p": 0.25,
                "better_decoder_by_mean": "logistic\rmodern",
            }
        ]
    )

    report = build_paired_stats_report(statistics)

    assert "\r" not in report
    assert "calibrated legacy" in report
    assert "lda legacy" in report
    assert "logistic modern" in report
