import pandas as pd
import pytest

from neureptrace.semantic_stages import summarize_category_timecourse


def _state_trace_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-a", "trial-b"],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "time": [0.1, 0.1],
            "true_class": ["animate", "inanimate"],
            "viterbi_class": ["animate", "inanimate"],
            "posterior_state_0": [0.9, 0.1],
            "posterior_state_1": [0.1, 0.9],
            "state_0": ["animate", "animate"],
            "state_1": ["inanimate", "inanimate"],
        }
    )


def test_semantic_stage_summary_rejects_state_columns_that_change_class():
    frame = _state_trace_frame()
    frame.loc[1, ["state_0", "state_1"]] = ["inanimate", "animate"]

    with pytest.raises(ValueError, match="state_0 values must identify one state"):
        summarize_category_timecourse(frame)


def test_semantic_stage_summary_rejects_duplicate_state_labels():
    frame = _state_trace_frame()
    frame["state_1"] = "animate"

    with pytest.raises(ValueError, match="State labels must map uniquely to posterior columns"):
        summarize_category_timecourse(frame)
