import pandas as pd

from neureptrace.semantic_stages import summarize_category_timecourse, summarize_dominant_timecourse


def test_semantic_stage_sequence_counts_preserve_structured_identity():
    state_traces = pd.DataFrame(
        {
            "decoder": ["d"] * 4,
            "emission_mode": ["calibrated"] * 4,
            "subject": ["s"] * 4,
            "session": ["session|one", "session", "numeric", "numeric"],
            "sequence_id": ["trial", "one|trial", 1, "1"],
            "time": [0.0] * 4,
            "true_class": ["A"] * 4,
            "viterbi_class": ["A"] * 4,
            "posterior_state_0": [1.0] * 4,
            "state_0": ["A"] * 4,
        }
    )

    category_summary, _ = summarize_category_timecourse(state_traces)
    dominant_summary = summarize_dominant_timecourse(state_traces.drop(columns="true_class"))

    assert category_summary.iloc[0].n_sequences == 4
    assert dominant_summary.iloc[0].n_sequences == 4
