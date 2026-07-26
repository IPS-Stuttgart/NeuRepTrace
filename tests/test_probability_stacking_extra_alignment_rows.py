import pandas as pd
import pytest

from neureptrace.probability_stacking import align_probability_cube


def test_align_probability_cube_rejects_extra_candidate_alignment_rows() -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "source",
                "sample_index": 0,
                "decoder": "reference",
                "true_label": 0,
                "true_class": "zero",
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "subject": "source",
                "sample_index": 1,
                "decoder": "reference",
                "true_label": 1,
                "true_class": "one",
                "prob_class_0": 0.1,
                "prob_class_1": 0.9,
            },
            {
                "subject": "source",
                "sample_index": 0,
                "decoder": "extra",
                "true_label": 0,
                "true_class": "zero",
                "prob_class_0": 0.8,
                "prob_class_1": 0.2,
            },
            {
                "subject": "source",
                "sample_index": 1,
                "decoder": "extra",
                "true_label": 1,
                "true_class": "one",
                "prob_class_0": 0.2,
                "prob_class_1": 0.8,
            },
            {
                "subject": "source",
                "sample_index": 2,
                "decoder": "extra",
                "true_label": 0,
                "true_class": "zero",
                "prob_class_0": 0.7,
                "prob_class_1": 0.3,
            },
        ]
    )

    with pytest.raises(ValueError, match="extra alignment-key row"):
        align_probability_cube(
            observations,
            candidates=("reference", "extra"),
            alignment_columns=("subject", "sample_index"),
        )
