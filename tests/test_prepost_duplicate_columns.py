import pandas as pd
import pytest

from neureptrace.metrics import summarize_window_metric


@pytest.mark.parametrize(
    ("columns", "row", "group_columns", "duplicate_name"),
    [
        (["time", "time", "accuracy"], [0.0, 0.1, 0.5], (), "time"),
        (["time", "accuracy", "accuracy"], [0.0, 0.5, 0.6], (), "accuracy"),
        (["time", "accuracy", "decoder", "decoder"], [0.0, 0.5, "lda", "logistic"], ("decoder",), "decoder"),
    ],
)
def test_summarize_window_metric_rejects_ambiguous_required_columns(
    columns: list[str],
    row: list[object],
    group_columns: tuple[str, ...],
    duplicate_name: str,
) -> None:
    frame = pd.DataFrame([row], columns=columns)

    with pytest.raises(ValueError, match=rf"ambiguous duplicate required columns.*{duplicate_name}"):
        summarize_window_metric(
            frame,
            "accuracy",
            (-0.1, 0.2),
            group_columns=group_columns,
        )
