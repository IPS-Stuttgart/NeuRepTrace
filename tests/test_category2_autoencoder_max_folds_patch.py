from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import neureptrace.bushmeg_category2_autoencoder_loso as category2_module
import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.dataset_config import load_config


def _write_minimal_category2_config(path: Path) -> None:
    path.write_text(
        """
participants:
  ids: "1,2,3"
preprocessing:
  window_size: 0.1
decoding:
  max_iter: 10
category2_autoencoder_loso:
  window_centers: [0.184]
  latent_dim: 2
""".lstrip(),
        encoding="utf-8",
    )


def test_category2_autoencoder_direct_runner_honors_max_folds_without_shrinking_source_pool(tmp_path, monkeypatch) -> None:
    features = {
        "1": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "2": np.asarray([[2.0, 2.0], [3.0, 3.0]], dtype=float),
        "3": np.asarray([[4.0, 4.0], [5.0, 5.0]], dtype=float),
    }
    subjects = {
        subject: SimpleNamespace(
            labels=np.asarray([0, 1], dtype=int),
            metadata=pd.DataFrame({"participant": [subject, subject], "stimulus_class": ["a", "b"]}),
        )
        for subject in ("1", "2", "3")
    }
    encoder = SimpleNamespace(classes_=np.asarray(["a", "b"], dtype=object))
    seen: dict[str, list[int]] = {"source_rows": [], "target_rows": []}

    class FakeFeatureCache:
        def __init__(self, loaded_subjects):
            self.loaded_subjects = loaded_subjects

        def get(self, subject_id, window, temporal_bins, *, feature_kind, covariance_max_channels):
            del window, temporal_bins, feature_kind, covariance_max_channels
            return features[str(subject_id)]

    def fake_load_subjects(config, *, config_dir):
        del config, config_dir
        return subjects, encoder

    def fake_fit_latents(x_source, x_target, cfg):
        del cfg
        seen["source_rows"].append(int(x_source.shape[0]))
        seen["target_rows"].append(int(x_target.shape[0]))
        return category2_module.AutoencoderFoldResult(
            z_source=x_source,
            z_target=x_target,
            reconstruction_mse_all=0.0,
            reconstruction_mse_source=0.0,
            reconstruction_mse_target=0.0,
            effective_latent_dim=int(x_source.shape[1]),
            n_autoencoder_iterations=None,
        )

    def fake_predict(z_source, y_source, z_target, cfg, *, n_classes):
        del z_source, y_source, cfg
        assert n_classes == 2
        probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=float)
        return probabilities[: z_target.shape[0]]

    monkeypatch.setattr(category2_module, "_load_subjects_from_config", fake_load_subjects)
    monkeypatch.setattr(category2_module, "FeatureCache", FakeFeatureCache)
    monkeypatch.setattr(category2_module, "_fit_autoencoder_latents", fake_fit_latents)
    monkeypatch.setattr(category2_module, "_predict_source_classifier", fake_predict)

    config_path = tmp_path / "category2.yml"
    summary_path = tmp_path / "summary.csv"
    predictions_path = tmp_path / "predictions.csv"
    _write_minimal_category2_config(config_path)

    summary = category2_module.run_bushmeg_category2_autoencoder_loso(
        config_path,
        out_path=summary_path,
        predictions_out_path=predictions_path,
        max_folds=1,
    )
    predictions = pd.read_csv(predictions_path)
    provenance = json.loads(Path(str(summary_path) + ".provenance.json").read_text(encoding="utf-8"))

    assert summary["outer_test_subject"].tolist() == ["1"]
    assert summary["n_train_subjects"].tolist() == [2]
    assert summary["max_folds"].tolist() == [1]
    assert summary["n_outer_folds"].tolist() == [1]
    assert predictions["outer_test_subject"].astype(str).unique().tolist() == ["1"]
    assert seen["source_rows"] == [4]
    assert seen["target_rows"] == [2]
    assert provenance["max_folds"] == 1
    assert provenance["n_outer_folds"] == 1


def test_category2_autoencoder_all_protocols_forwards_fold_limit_to_method_config(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_category2(config_path, *, out_path=None, predictions_out_path=None):
        config_path = Path(config_path)
        config = load_config(config_path)
        seen["config_path"] = config_path
        seen["participants"] = config["participants"]["ids"]
        seen["max_folds"] = config["category2_autoencoder_loso"]["max_folds"]

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
    assert seen["max_folds"] == 1
    assert seen["config_path"] == method_dir / "config.yml"
    assert result.method_metadata.loc[0, "method"] == "category2_autoencoder_loso"
    assert result.summary["method"].tolist() == ["category2_autoencoder_loso"]
