from pathlib import Path

import pandas as pd
import pytest

from neureptrace.temporal_model import fit_temporal_models


def _observation_frame(*, baseline_times: tuple[float, ...] = (-0.08, -0.04)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence_id in range(3):
        for time in (*baseline_times, 0.10, 0.20):
            p0 = 0.9 if time <= 0.10 else 0.1
            rows.append(
                {
                    "subject": "sub-01",
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "sequence_id": sequence_id,
                    "time": time,
                    "prob_class_0": p0,
                    "prob_class_1": 1.0 - p0,
                }
            )
    return pd.DataFrame(rows)


def test_fit_temporal_models_rejects_duplicate_baseline_times(tmp_path: Path) -> None:
    frame = _observation_frame()
    duplicate = frame.loc[frame["sequence_id"].eq(0) & frame["time"].eq(-0.08)].copy()
    csv_path = tmp_path / "duplicate_baseline.csv"
    pd.concat([frame, duplicate], ignore_index=True).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Duplicate time rows found within a sequence identity"):
        fit_temporal_models(
            [csv_path],
            effect_window=(0.10, 0.20),
            baseline_window=(-0.10, 0.0),
            n_permutations=0,
            stay_grid_size=5,
        )


def test_fit_temporal_models_still_allows_singleton_baselines(tmp_path: Path) -> None:
    csv_path = tmp_path / "singleton_baseline.csv"
    _observation_frame(baseline_times=(-0.08,)).to_csv(csv_path, index=False)

    summary, states = fit_temporal_models(
        [csv_path],
        effect_window=(0.10, 0.20),
        baseline_window=(-0.10, 0.0),
        n_permutations=0,
        stay_grid_size=5,
    )

    assert states is None
    assert summary["condition"].tolist() == ["observed_effect"]
