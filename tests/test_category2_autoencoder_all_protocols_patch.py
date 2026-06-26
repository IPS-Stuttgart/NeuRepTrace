from __future__ import annotations

import json
from pathlib import Path

import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.dataset_config import load_config

from _category2_autoencoder_fakes import category2_autoencoder_result_frames


def test_category2_autoencoder_loso_is_runnable_all_protocol_method() -> None:
    spec = all_protocols.method_registry()["category2_autoencoder_loso"]

    assert spec.protocol_category == 2
    assert spec.method_family == "category2_autoencoder"
    assert spec.runner == "category2_autoencoder_loso"
    assert spec.runnable is True
    assert spec.metadata()["inventory_only"] is False


def test_category2_autoencoder_loso_is_available_when_module_is_present(monkeypatch) -> None:
    spec = all_protocols.method_registry()["category2_autoencoder_loso"]
    monkeypatch.setattr(
        all_protocols,
        "_module_available",
        lambda module: module == "neureptrace.bushmeg_category2_autoencoder_loso",
    )

    available, skip_reason = all_protocols._method_availability(
        spec,
        {},
        settings={},
        include_heavy=False,
        max_folds=1,
    )

    assert available is True
    assert skip_reason == ""


def test_category2_autoencoder_all_protocols_runner_writes_progress_artifacts(tmp_path, monkeypatch) -> None:
    import neureptrace.bushmeg_category2_autoencoder_loso as category2_module

    seen: dict[str, object] = {}

    def fake_run_category2(config_path, *, out_path=None, predictions_out_path=None):
        config_path = Path(config_path)
        config = load_config(config_path)
        seen["config_path"] = config_path
        seen["participants"] = config["participants"]["ids"]

        summary, predictions = category2_autoencoder_result_frames()
        assert out_path is not None
        assert predictions_out_path is not None
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        predictions.to_csv(predictions_out_path, index=False)
        return summary

    monkeypatch.setattr(category2_module, "run_bushmeg_category2_autoencoder_loso", fake_run_category2)

    result = all_protocols.run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="category2_autoencoder_loso",
        protocols="2",
        participants="1,2,3",
        fold_limit=1,
        resume=False,
    )

    method_dir = tmp_path / "methods" / "category2_autoencoder_loso"
    assert seen["participants"] == "1,2,3"
    assert seen["config_path"] == method_dir / "config.yml"
    assert (method_dir / "summary.partial.csv").exists()
    assert (method_dir / "predictions.partial.csv").exists()
    assert (method_dir / "inner_cv.partial.csv").exists()
    assert (method_dir / "summary.csv").exists()
    assert (method_dir / "predictions.csv").exists()
    assert (method_dir / "inner_cv.csv").exists()

    status = json.loads((method_dir / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "method_done"
    assert status["method"] == "category2_autoencoder_loso"
    assert status["n_summary_rows"] == 1
    assert status["n_prediction_rows"] == 2

    log_text = (method_dir / "run.log").read_text(encoding="utf-8")
    for stage in ("configured", "checking_requirements", "loading_subjects", "method_done"):
        assert f'"stage": "{stage}"' in log_text

    assert result.method_metadata.loc[0, "method"] == "category2_autoencoder_loso"
    assert result.method_metadata.loc[0, "status"] == "runnable"
    assert result.summary["method"].tolist() == ["category2_autoencoder_loso"]
    assert result.predictions["method"].tolist() == [
        "category2_autoencoder_loso",
        "category2_autoencoder_loso",
    ]
