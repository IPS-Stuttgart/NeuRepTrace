import pandas as pd

from neureptrace.paired_stats import build_paired_stats_report, paired_decoder_statistics


def test_paired_stats_report_preserves_nondefault_condition_columns() -> None:
    rows = []
    for pca_components, lda_accuracy, logistic_accuracy in [("10", 0.50, 0.70), ("20", 0.80, 0.60)]:
        for subject in ["sub-01", "sub-02"]:
            rows.append(
                {
                    "emission_mode": "calibrated",
                    "pca_components": pca_components,
                    "decoder": "lda",
                    "subject": subject,
                    "effect_accuracy": lda_accuracy,
                }
            )
            rows.append(
                {
                    "emission_mode": "calibrated",
                    "pca_components": pca_components,
                    "decoder": "logistic",
                    "subject": subject,
                    "effect_accuracy": logistic_accuracy,
                }
            )

    stats = paired_decoder_statistics(pd.DataFrame(rows), metrics=("effect_accuracy",), n_permutations=10_000)
    report = build_paired_stats_report(stats)

    assert "| Emission mode | PCA components | Decoder A |" in report
    assert "| calibrated | 10 | lda | logistic | effect_accuracy |" in report
    assert "| calibrated | 20 | lda | logistic | effect_accuracy |" in report
