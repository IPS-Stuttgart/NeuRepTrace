from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from neureptrace.loso_observation_diagnostics import write_loso_observation_diagnostics


def test_loso_diagnostics_preserve_adjacent_large_integer_labels(tmp_path: Path) -> None:
    labels = (2**53, 2**53 + 1, 2**53 + 2)
    rows: list[dict[str, object]] = []
    for subject in ("sub-01", "sub-02"):
        for sample_index, label in enumerate(labels):
            row: dict[str, object] = {
                "group": subject,
                "time": 0.1,
                "sample_index": f"{subject}-{sample_index}",
                "true_label": label,
                "predicted_label": label,
                "true_class": f"class_{label}",
                "predicted_class": f"class_{label}",
                "confidence": 0.8,
            }
            for probability_label in labels:
                row[f"prob_class_{probability_label}"] = 0.8 if probability_label == label else 0.1
            rows.append(row)

    observations_csv = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(observations_csv, index=False)

    paths = write_loso_observation_diagnostics(
        observations_csv,
        out_dir=tmp_path / "diagnostics",
        best_time=0.1,
    )

    quality = pd.read_csv(paths["quality_summary"])
    selective = pd.read_csv(paths["selective_coverage"])
    full_coverage = selective.loc[selective["coverage_target"].eq(1.0)].iloc[0]

    assert quality.loc[0, "fixed_log_loss"] == pytest.approx(-math.log(0.8))
    assert bool(full_coverage["all_classes_present"])
    assert json.loads(full_coverage["selected_class_support"]) == {
        str(label): 2 for label in labels
    }
