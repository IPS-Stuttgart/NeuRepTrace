from __future__ import annotations

from pathlib import Path

import pandas as pd

from neureptrace import benchmark


def test_manifest_passes_normalization_and_baseline_window(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,label_column,normalization,baseline_window_start,baseline_window_stop\n"
        "S1,s1-epo.fif,condition,subject-baseline-whiten,-0.5,0.0\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_time_resolved_decode(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "time": 0.0,
                    "accuracy": 1.0,
                    "log_loss": 0.0,
                    "brier": 0.0,
                    "ece": 0.0,
                }
            ]
        )

    monkeypatch.setattr(benchmark, "run_time_resolved_decode", fake_run_time_resolved_decode)
    monkeypatch.setattr(
        benchmark,
        "aggregate_time_decode_csvs",
        lambda *_args, **_kwargs: pd.DataFrame({"subject": ["S1"], "accuracy": [1.0]}),
    )
    monkeypatch.setattr(
        benchmark,
        "write_provenance_table",
        lambda *_args, **_kwargs: pd.DataFrame({"n_subjects": [1]}),
    )

    run = benchmark.run_benchmark_manifest(
        manifest,
        out_dir=tmp_path / "out",
    )

    assert run.result_csvs == [tmp_path / "out" / "S1_subject_baseline_whiten_base-0p5_0_time_decode.csv"]
    assert captured["normalization"] == "subject_baseline_whiten"
    assert captured["baseline_window"] == (-0.5, 0.0)


def test_manifest_accepts_compact_baseline_window(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,label_column,baseline_window\n"
        "S1,s1-epo.fif,condition,-0.25|0.05\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_time_resolved_decode(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "time": 0.0,
                    "accuracy": 1.0,
                    "log_loss": 0.0,
                    "brier": 0.0,
                    "ece": 0.0,
                }
            ]
        )

    monkeypatch.setattr(benchmark, "run_time_resolved_decode", fake_run_time_resolved_decode)
    monkeypatch.setattr(
        benchmark,
        "aggregate_time_decode_csvs",
        lambda *_args, **_kwargs: pd.DataFrame({"subject": ["S1"], "accuracy": [1.0]}),
    )
    monkeypatch.setattr(
        benchmark,
        "write_provenance_table",
        lambda *_args, **_kwargs: pd.DataFrame({"n_subjects": [1]}),
    )

    benchmark.run_benchmark_manifest(
        manifest,
        out_dir=tmp_path / "out",
        default_normalization="subject_baseline_z",
    )

    assert captured["normalization"] == "subject_baseline_z"
    assert captured["baseline_window"] == (-0.25, 0.05)


def test_manifest_requires_complete_split_baseline_window(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,label_column,baseline_window_start\n"
        "S1,s1-epo.fif,condition,-0.5\n",
        encoding="utf-8",
    )

    try:
        benchmark.run_benchmark_manifest(manifest, out_dir=tmp_path / "out")
    except ValueError as exc:
        assert "both baseline_window_start and baseline_window_stop" in str(exc)
    else:  # pragma: no cover - defensive assertion for pytest without pytest.raises.
        raise AssertionError("expected incomplete split baseline window to fail")
