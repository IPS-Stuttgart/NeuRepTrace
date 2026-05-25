from pathlib import Path

import pandas as pd
import pytest

from neureptrace.benchmark import run_benchmark_manifest


def _fake_decode(**kwargs):
    frame = pd.DataFrame(
        {
            "fold": [0],
            "decoder": [kwargs.get("decoder", "logistic")],
            "emission_mode": [kwargs.get("emission_mode", "calibrated")],
            "time": [0.1],
            "accuracy": [0.75],
            "log_loss": [0.4],
            "brier": [0.2],
            "ece": [0.1],
        }
    )
    out_path = kwargs["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def _write_manifest(tmp_path: Path, extra_columns: list[str], extra_values: list[str]) -> Path:
    manifest = tmp_path / "manifest.csv"
    columns = ["subject", "epochs", "metadata_csv", "label_column", *extra_columns]
    values = [
        "sub-01",
        "data/sub-01_epo.fif",
        "data/sub-01_metadata.csv",
        "condition",
        *extra_values,
    ]
    manifest.write_text(
        ",".join(columns) + "\n" + ",".join(values) + "\n",
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize(
    "column",
    ["n_splits", "max_iter", "calibration_bins", "tuning_cv_splits"],
)
def test_run_benchmark_manifest_rejects_fractional_integer_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
):
    manifest = _write_manifest(tmp_path, [column], ["2.5"])
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("neureptrace.benchmark.run_time_resolved_decode", fake_decode)

    with pytest.raises(ValueError, match=f"'{column}'"):
        run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert calls == []


def test_run_benchmark_manifest_accepts_integer_like_integer_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = _write_manifest(
        tmp_path,
        ["n_splits", "max_iter", "calibration_bins", "tuning_cv_splits"],
        ["2.0", "100.0", "5.0", "3.0"],
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("neureptrace.benchmark.run_time_resolved_decode", fake_decode)

    run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert len(calls) == 1
    assert calls[0]["n_splits"] == 2
    assert calls[0]["max_iter"] == 100
    assert calls[0]["calibration_bins"] == 5
    assert calls[0]["tuning_cv_splits"] == 3
