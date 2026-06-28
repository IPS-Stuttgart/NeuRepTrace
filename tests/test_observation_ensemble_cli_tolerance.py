from __future__ import annotations

from pathlib import Path

import pandas as pd

import neureptrace.observation_ensemble as observation_ensemble


class _DummyObservationTable:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def to_csv(self, path: Path) -> None:
        path.write_text("ok\n", encoding="utf-8")


def test_observation_ensemble_cli_forwards_probability_tolerance(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, float] = {}
    observations = pd.DataFrame(
        {
            "decoder": ["logistic", "linear_svm"],
            "time": [0.0, 0.0],
            "true_label": [0, 0],
            "prob_class_0": [0.55, 0.55],
            "prob_class_1": [0.50, 0.50],
        }
    )

    def fake_read_validated_probability_observations(observation_csv, **kwargs):
        seen["reader_probability_tolerance"] = kwargs["probability_tolerance"]
        return observations

    def fake_ensemble(observations, **kwargs):
        seen["ensemble_probability_tolerance"] = kwargs["probability_tolerance"]
        return observations

    monkeypatch.setattr(observation_ensemble, "read_validated_probability_observations", fake_read_validated_probability_observations)
    monkeypatch.setattr(observation_ensemble, "ensemble_probability_observations", fake_ensemble)
    monkeypatch.setattr(observation_ensemble, "ProbabilityObservationTable", _DummyObservationTable)

    out_path = tmp_path / "ensemble.csv"
    exit_code = observation_ensemble.main(
        [
            "observations.csv",
            "--out",
            str(out_path),
            "--probability-tolerance",
            "0.1",
        ]
    )

    assert exit_code == 0
    assert out_path.read_text(encoding="utf-8") == "ok\n"
    assert seen["reader_probability_tolerance"] == 0.1
    assert seen["ensemble_probability_tolerance"] == 0.1
