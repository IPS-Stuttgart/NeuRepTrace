from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.dataset_config import load_config


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

        summary = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "balanced_accuracy": 1.0,
                    "accuracy": 1.0,
                    "top2_accuracy": 1.0,
                    "top3_accuracy": 1.0,
                    "log_loss": 0.1,
                    "brier": 0.05,
                    "ece": 0.0,
                    "n_train_subjects": 2,
                    "n_source_trials": 4,
                    "n_target_trials": 2,
                    "n_classes": 2,
                    "class_names": "0|1",
                    "feature_kind": "evoked_dct",
                    "temporal_bins": 4,
                    "window_centers": "0.184",
                    "window_widths": "0.1",
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "trial_index": 0,
                    "true_label": 0,
                    "predicted_label": 0,
                    "prob_class_0": 0.9,
                    "prob_class_1": 0.1,
                },
                {
                    "outer_test_subject": "1",
                    "trial_index": 1,
                    "true_label": 1,
                    "predicted_label": 1,
                    "prob_class_0": 0.2,
                    "prob_class_1": 0.8,
                },
            ]
        )
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
