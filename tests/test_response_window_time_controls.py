from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.response_window_ensemble import run_response_window_ensemble


def _write_minimal_observations(path: Path) -> None:
    rows = []
    for sample_index, true_label in enumerate((0, 1)):
        for time in (0.088, 0.184):
            probabilities = np.array([0.8, 0.2]) if true_label == 0 else np.array([0.2, 0.8])
            predicted_label = int(probabilities.argmax())
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": "sub-01",
                    "decoder": "base",
                    "emission_mode": "calibrated",
                    "time": time,
                    "test_time": time,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class-{predicted_label}",
                    "probability_true_class": float(probabilities[true_label]),
                    "confidence": float(probabilities.max()),
                    "class_0": "class-0",
                    "class_1": "class-1",
                    "prob_class_0": float(probabilities[0]),
                    "prob_class_1": float(probabilities[1]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"response_times": (np.asarray(True),)}, "times must be finite"),
        ({"response_times": (np.array([0.088]),)}, "times must be finite"),
        ({"output_time": np.asarray(True)}, "output_time must be finite"),
        ({"output_time": np.array([0.184])}, "output_time must be finite"),
    ],
)
def test_response_window_rejects_array_valued_time_controls(tmp_path: Path, kwargs: dict[str, object], message: str):
    csv_path = tmp_path / "observations.csv"
    _write_minimal_observations(csv_path)

    with pytest.raises(ValueError, match=message):
        run_response_window_ensemble([csv_path], mode="uniform", **kwargs)
