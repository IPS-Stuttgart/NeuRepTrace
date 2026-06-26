from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.dataset_config import load_config
from neureptrace.bushmeg_all_protocols import (
    PROTOCOLS,
    MethodSpec,
    MethodProgress,
    RunTimeoutError,
    _selected_methods,
    build_registry_audit,
    category3_calibration_evaluation_split,
    main,
    method_registry,
    rebuild_all_protocol_outputs_from_partials,
    run_bushmeg_all_protocols,
    run_bushmeg_protocol3_fold_adapter,
    select_bushmeg_target_calibration_split,
    validate_disjoint_calibration_evaluation,
    validate_protocol_input_use,
    validate_target_label_policy,
)


def _install_synthetic_missing_method(monkeypatch: pytest.MonkeyPatch) -> str:
    method = "synthetic_missing_module_method"
    registry = dict(method_registry())
    registry[method] = MethodSpec(
        method,
        "synthetic_missing_module_family",
        1,
        "unavailable",
        runnable=False,
        blocked_reason="inventory-only synthetic method for registry audit tests",
        required_modules=("neureptrace.decoding.definitely_missing_registry_audit_test",),
    )
    monkeypatch.setattr(all_protocols, "method_registry", lambda *args, **kwargs: registry)
    return method


def test_protocol_metadata_correctness() -> None:
    strict = PROTOCOLS[1].metadata()
    assert strict["protocol_name"] == "strict_source_only"
    assert strict["uses_source_data"] is True
    assert strict["uses_source_labels"] is True
    assert strict["uses_target_data"] is False
    assert strict["uses_target_labels_for_fitting"] is False
    assert strict["calibration_rows_disjoint_from_evaluation"] is True
    assert strict["valid_for_strict_source_only"] is True
    assert strict["valid_for_zero_calibration"] is True
    assert strict["debug_upper_bound"] is False
    assert strict["uses_target_labels_for_scoring_only"] is True
    assert strict["target_data_use"] == "inference_only"

    unlabeled = PROTOCOLS[2].metadata()
    assert unlabeled["protocol_name"] == "unlabeled_transductive_adaptation"
    assert unlabeled["uses_target_data"] is True
    assert unlabeled["uses_target_labels_for_fitting"] is False
    assert unlabeled["valid_for_strict_source_only"] is False
    assert unlabeled["valid_for_zero_calibration"] is True
    assert unlabeled["debug_upper_bound"] is False
    assert unlabeled["target_data_use"] == "unlabeled_adaptation_and_inference"

    registry = method_registry()
    assert registry["source_alignment_procrustes_group_projection"].protocol_category == 1
    assert registry["coral_alignment"].protocol_category == 2
    assert registry["target_calibrated_mcca"].protocol_category == 3
    assert registry["oracle_target_calibrated_mcca"].protocol_category == 4


def test_method_registry_covers_discussed_method_ids() -> None:
    expected = {
        "source_loso_logistic",
        "source_loso_linear_svm",
        "source_loso_correlation_prototype",
        "source_loso_decoder_ensemble",
        "source_probability_calibration_none",
        "source_probability_calibration_temperature",
        "source_probability_calibration_class_bias",
        "source_probability_calibration_temperature_plus_class_bias",
        "source_probability_calibration_confusion_correction_l2",
        "source_alignment_procrustes_group_projection",
        "source_alignment_hyperalignment_group_projection",
        "source_alignment_mcca_group_projection",
        "contrastive_group_projection",
        "source_domain_generalization_erm",
        "source_domain_generalization_subject_adversarial",
        "source_domain_generalization_group_dro",
        "reconstruction_source_only",
        "generative_source_gaussian",
        "generative_source_gan",
        "generative_source_diffusion",
        "foundation_frozen_linear_probe",
        "euclidean_alignment",
        "coral_alignment",
        "target_baseline_covariance",
        "subject_sensor_covariance",
        "sinkhorn_transport",
        "group_projection_target_centered",
        "pseudo_label_target_calibrated_alignment",
        "pseudo_label_self_training",
        "riemannian_tangent_transfer",
        "riemannian_procrustes_no_rotation",
        "riemannian_procrustes_paired_unlabeled",
        "mekt",
        "optimal_transport_sinkhorn",
        "source_weighting_target_similarity",
        "source_weighting_hybrid",
        "dann",
        "cdan",
        "cdan_mmd",
        "cdan_conditional_mmd",
        "ttime_after_predict",
        "ttime_before_predict",
        "source_free_adaptation",
        "reconstruction_source_plus_target",
        "unlabeled_calibration_hyperalignment",
        "unlabeled_calibration_mcca",
        "unlabeled_calibration_procrustes",
        "target_style_gaussian",
        "target_style_gan",
        "target_style_diffusion",
        "weak_label_proportion_calibration",
        "target_calibrated_procrustes",
        "target_calibrated_hyperalignment",
        "target_calibrated_mcca",
        "contrastive_target_calibrated",
        "few_shot_target_calibrated_decoder_k1",
        "few_shot_target_calibrated_decoder_k2",
        "few_shot_target_calibrated_decoder_k4",
        "few_shot_target_calibrated_decoder_k8",
        "few_shot_target_calibrated_decoder_k16",
        "semi_supervised_lora_few_shot_k1",
        "semi_supervised_lora_few_shot_k2",
        "semi_supervised_lora_few_shot_k4",
        "semi_supervised_lora_few_shot_k8",
        "semi_supervised_lora_few_shot_k16",
        "target_calibrated_gaussian",
        "target_calibrated_gan",
        "target_calibrated_diffusion",
        "oracle_target_calibrated_procrustes",
        "oracle_target_calibrated_hyperalignment",
        "oracle_target_calibrated_mcca",
    }
    assert expected.issubset(method_registry())


def test_few_shot_target_calibrated_decoder_k_methods_are_runnable() -> None:
    registry = method_registry()
    for k in (1, 2, 4, 8, 16):
        spec = registry[f"few_shot_target_calibrated_decoder_k{k}"]
        assert spec.protocol_category == 3
        assert spec.method_family == "few_shot_target_calibration"
        assert spec.runner == "protocol3_few_shot"
        assert spec.runnable is True
        assert spec.config_updates["protocol3"]["target_calibration_per_class"] == k


@pytest.mark.parametrize("protocol", [1, 2])
def test_target_labels_rejected_in_protocol_1_and_2_methods(protocol: int) -> None:
    with pytest.raises(ValueError, match="must not use held-out target labels"):
        validate_target_label_policy(protocol, uses_target_labels_for_fitting=True)

    validate_target_label_policy(protocol, uses_target_labels_for_fitting=False)


def test_protocol1_rejects_target_features_for_fitting() -> None:
    with pytest.raises(ValueError, match="Protocol 1 .*target features"):
        validate_protocol_input_use(1, target_features_for_fitting=True)

    validate_protocol_input_use(1, target_features_for_fitting=False)


@pytest.mark.parametrize(
    ("leakage_kwargs", "message"),
    [
        ({"target_labels_for_fitting": True}, "Protocol 2 .*target labels"),
        ({"target_class_prototypes_for_fitting": True}, "Protocol 2 .*target class prototypes"),
        ({"target_accuracy_for_model_selection": True}, "Protocol 2 .*target accuracy"),
    ],
)
def test_protocol2_rejects_target_supervision_for_fitting(leakage_kwargs: dict[str, bool], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_protocol_input_use(2, target_features_for_fitting=True, **leakage_kwargs)

    validate_protocol_input_use(2, target_features_for_fitting=True)


def test_select_bushmeg_target_calibration_split_k1_selects_one_row_per_class() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=1,
        seed=13,
        context=("sub-01", "few_shot_target_calibrated_decoder", 1),
    )

    assert split.skipped is False
    assert split.calibration_indices.size == 3
    assert split.evaluation_indices.size == 6
    for class_value in np.unique(labels):
        assert np.count_nonzero(labels[split.calibration_indices] == class_value) == 1
        assert np.count_nonzero(labels[split.evaluation_indices] == class_value) >= 1
    metadata = split.metadata()
    assert metadata["protocol_category"] == 3
    assert metadata["target_calibration_per_class"] == 1
    assert metadata["n_target_calibration_trials"] == 3
    assert metadata["n_target_evaluation_trials"] == 6
    assert metadata["calibration_rows_disjoint_from_evaluation"] is True
    assert metadata["target_calibration_seed"] == 13
    assert metadata["uses_target_data"] is True
    assert metadata["uses_target_labels_for_fitting"] is True
    assert metadata["valid_for_zero_calibration"] is False
    assert metadata["valid_for_strict_source_only"] is False


def test_select_bushmeg_target_calibration_split_evaluation_rows_are_disjoint() -> None:
    labels = np.asarray(["face", "face", "face", "scrambled", "scrambled", "scrambled"])

    split = select_bushmeg_target_calibration_split(labels, per_class=1, seed=9, context=("sub-02", "method", 1))

    assert split.skipped is False
    assert np.intersect1d(split.calibration_indices, split.evaluation_indices).size == 0
    assert sorted(np.concatenate([split.calibration_indices, split.evaluation_indices]).tolist()) == list(range(labels.size))


def test_select_bushmeg_target_calibration_split_is_deterministic() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    kwargs = {"per_class": 2, "seed": 21, "context": ("sub-03", "target_calibrated_mcca", 2)}

    first = select_bushmeg_target_calibration_split(labels, **kwargs)
    second = select_bushmeg_target_calibration_split(labels, **kwargs)

    assert first.skipped is False
    assert np.array_equal(first.calibration_indices, second.calibration_indices)
    assert np.array_equal(first.evaluation_indices, second.evaluation_indices)
    assert first.effective_seed == second.effective_seed


def test_select_bushmeg_target_calibration_split_insufficient_rows_returns_skip_reason() -> None:
    labels = np.asarray([0, 1, 1, 2, 2])

    split = select_bushmeg_target_calibration_split(labels, per_class=1, seed=13, context=("sub-04", "method", 1))

    assert split.skipped is True
    assert split.skip_reason_code == "insufficient_rows_per_class"
    assert "class 0" in split.skip_reason
    assert split.calibration_indices.size == 0
    assert split.evaluation_indices.size == 0
    assert split.metadata()["target_calibration_skipped"] is True


def test_select_bushmeg_target_calibration_split_does_not_consume_any_evaluation_class() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2])

    split = select_bushmeg_target_calibration_split(labels, per_class=1, seed=5, context=("sub-05", "method", 1))

    assert split.skipped is False
    for class_value in np.unique(labels):
        assert np.count_nonzero(labels[split.evaluation_indices] == class_value) == 1

    infeasible = select_bushmeg_target_calibration_split(labels, per_class=2, seed=5, context=("sub-05", "method", 2))
    assert infeasible.skipped is True
    assert infeasible.skip_reason_code == "insufficient_rows_per_class"


def _protocol3_adapter_fixture(tmp_path):
    method_spec = MethodSpec("synthetic_protocol3_method", "synthetic_protocol3", 3, "protocol3_fold_adapter")
    source_labels = np.asarray([0, 1, 2, 0, 1, 2])
    source_features = np.column_stack([source_labels, np.arange(source_labels.size)])
    source_subject_ids = np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"])
    target_labels = np.asarray([0, 0, 1, 1, 2, 2])
    target_features = np.column_stack([target_labels, np.arange(target_labels.size)])
    classes = np.asarray([0, 1, 2])
    method_dir = tmp_path / "methods" / method_spec.method
    return method_spec, source_features, source_labels, source_subject_ids, target_features, target_labels, classes, method_dir


def _feature_label_probability_predictor(**kwargs):
    evaluation_features = np.asarray(kwargs["target_evaluation_features"])
    classes = np.asarray(kwargs["classes"])
    class_values = evaluation_features[:, 0].astype(int)
    probabilities = np.zeros((evaluation_features.shape[0], classes.size), dtype=float)
    for row_index, class_value in enumerate(class_values):
        class_index = int(np.flatnonzero(classes == class_value)[0])
        probabilities[row_index, class_index] = 1.0
    return {"probabilities": probabilities}


def test_protocol3_fold_adapter_does_not_pass_evaluation_labels_into_fitting(tmp_path) -> None:
    (
        method_spec,
        source_features,
        source_labels,
        source_subject_ids,
        target_features,
        target_labels,
        classes,
        method_dir,
    ) = _protocol3_adapter_fixture(tmp_path)
    seen = {}

    def fake_fit_predict(**kwargs):
        seen.update(kwargs)
        return _feature_label_probability_predictor(**kwargs)

    run_bushmeg_protocol3_fold_adapter(
        source_features=source_features,
        source_labels=source_labels,
        source_subject_ids=source_subject_ids,
        target_features=target_features,
        target_labels=target_labels,
        classes=classes,
        method_spec=method_spec,
        k_per_class=1,
        method_dir=method_dir,
        fit_predict=fake_fit_predict,
        outer_test_subject="target-01",
        seed=13,
    )

    assert "target_evaluation_labels" not in seen
    assert "target_labels" not in seen
    assert "target_label_vector" not in seen
    assert "target_calibration_labels" in seen
    assert len(seen["target_calibration_labels"]) == 3
    assert set(seen["target_calibration_labels"].tolist()) == {0, 1, 2}


def test_protocol3_fold_adapter_excludes_calibration_rows_from_predictions(tmp_path) -> None:
    (
        method_spec,
        source_features,
        source_labels,
        source_subject_ids,
        target_features,
        target_labels,
        classes,
        method_dir,
    ) = _protocol3_adapter_fixture(tmp_path)

    result = run_bushmeg_protocol3_fold_adapter(
        source_features=source_features,
        source_labels=source_labels,
        source_subject_ids=source_subject_ids,
        target_features=target_features,
        target_labels=target_labels,
        classes=classes,
        method_spec=method_spec,
        k_per_class=1,
        method_dir=method_dir,
        fit_predict=_feature_label_probability_predictor,
        outer_test_subject="target-01",
        seed=13,
    )

    predicted_rows = set(result.predictions["trial_index"].astype(int).tolist())
    calibration_rows = set(result.split.calibration_indices.tolist())
    evaluation_rows = set(result.split.evaluation_indices.tolist())
    assert predicted_rows == evaluation_rows
    assert predicted_rows.isdisjoint(calibration_rows)
    assert result.predictions["is_calibration_row"].eq(False).all()


def test_protocol3_fold_adapter_metrics_are_computed_only_on_evaluation_rows(tmp_path) -> None:
    (
        method_spec,
        source_features,
        source_labels,
        source_subject_ids,
        target_features,
        target_labels,
        classes,
        method_dir,
    ) = _protocol3_adapter_fixture(tmp_path)

    result = run_bushmeg_protocol3_fold_adapter(
        source_features=source_features,
        source_labels=source_labels,
        source_subject_ids=source_subject_ids,
        target_features=target_features,
        target_labels=target_labels,
        classes=classes,
        method_spec=method_spec,
        k_per_class=1,
        method_dir=method_dir,
        fit_predict=_feature_label_probability_predictor,
        outer_test_subject="target-01",
        seed=13,
    )

    row = result.summary.iloc[0]
    assert int(row["n_test_trials"]) == result.split.evaluation_indices.size
    assert int(row["n_target_evaluation_trials"]) == result.split.evaluation_indices.size
    assert int(row["n_target_calibration_trials"]) == result.split.calibration_indices.size
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert result.predictions.shape[0] == result.split.evaluation_indices.size


def test_protocol3_fold_adapter_writes_partial_outputs_with_split_counts(tmp_path) -> None:
    (
        method_spec,
        source_features,
        source_labels,
        source_subject_ids,
        target_features,
        target_labels,
        classes,
        method_dir,
    ) = _protocol3_adapter_fixture(tmp_path)

    result = run_bushmeg_protocol3_fold_adapter(
        source_features=source_features,
        source_labels=source_labels,
        source_subject_ids=source_subject_ids,
        target_features=target_features,
        target_labels=target_labels,
        classes=classes,
        method_spec=method_spec,
        k_per_class=1,
        method_dir=method_dir,
        fit_predict=_feature_label_probability_predictor,
        outer_test_subject="target-01",
        seed=13,
    )

    summary_partial = pd.read_csv(method_dir / "summary.partial.csv")
    predictions_partial = pd.read_csv(method_dir / "predictions.partial.csv")
    status = json.loads((method_dir / "status.json").read_text(encoding="utf-8"))
    assert summary_partial.loc[0, "k_per_class"] == 1
    assert summary_partial.loc[0, "target_calibration_per_class"] == 1
    assert summary_partial.loc[0, "n_target_calibration_trials"] == result.split.calibration_indices.size
    assert summary_partial.loc[0, "n_target_evaluation_trials"] == result.split.evaluation_indices.size
    assert predictions_partial.shape[0] == result.split.evaluation_indices.size
    assert predictions_partial["k_per_class"].eq(1).all()
    assert status["stage"] == "fold_done"
    assert status["k_per_class"] == 1
    assert status["n_target_calibration_trials"] == result.split.calibration_indices.size
    assert status["n_target_evaluation_trials"] == result.split.evaluation_indices.size


def test_few_shot_protocol3_k_infeasible_fold_is_skipped(tmp_path) -> None:
    (
        _method_spec,
        source_features,
        source_labels,
        source_subject_ids,
        target_features,
        target_labels,
        classes,
        method_dir,
    ) = _protocol3_adapter_fixture(tmp_path)
    method_spec = method_registry()["few_shot_target_calibrated_decoder_k2"]

    result = run_bushmeg_protocol3_fold_adapter(
        source_features=source_features,
        source_labels=source_labels,
        source_subject_ids=source_subject_ids,
        target_features=target_features,
        target_labels=target_labels,
        classes=classes,
        method_spec=method_spec,
        k_per_class=2,
        method_dir=method_dir,
        fit_predict=_feature_label_probability_predictor,
        outer_test_subject="target-01",
        seed=13,
    )

    assert result.skipped is True
    assert result.split.skip_reason_code == "insufficient_rows_per_class"
    assert result.predictions.empty
    summary_partial = pd.read_csv(method_dir / "summary.partial.csv")
    assert summary_partial.loc[0, "skip_reason_code"] == "insufficient_rows_per_class"
    assert summary_partial.loc[0, "k_per_class"] == 2


def test_few_shot_fit_predict_probability_rows_align_to_all_classes() -> None:
    candidate = SimpleNamespace(
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
    )
    source_labels = np.asarray([0, 1, 2, 0, 1, 2])
    source_features = np.column_stack(
        [
            source_labels == 0,
            source_labels == 1,
            source_labels == 2,
            np.arange(source_labels.size) / 10.0,
        ]
    ).astype(float)
    target_calibration_labels = np.asarray([0, 1, 2])
    target_calibration_features = np.eye(3, 4, dtype=float)
    target_evaluation_features = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.7],
            [0.0, 1.0, 0.0, 0.8],
            [0.0, 0.0, 1.0, 0.9],
        ]
    )

    output = all_protocols._few_shot_target_calibrated_fit_predict(
        source_features=source_features,
        source_labels=source_labels,
        target_calibration_features=target_calibration_features,
        target_calibration_labels=target_calibration_labels,
        target_evaluation_features=target_evaluation_features,
        classes=np.asarray([0, 1, 2]),
        method_spec=method_registry()["few_shot_target_calibrated_decoder_k1"],
        k_per_class=1,
        candidate=candidate,
        decoding={"max_iter": 500},
        protocol3={"target_calibration_seed": 13, "target_repeats": 1},
    )

    probabilities = output["probabilities"]
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probabilities))


def test_protocol3_source_plus_alignment_gaussian_and_lora_k_methods_are_registered() -> None:
    registry = method_registry()
    for k in (1, 2, 4, 8, 16):
        assert registry[f"source_plus_target_calibration_logistic_k{k}"].runner == "protocol3_source_plus_target"
        assert registry[f"source_plus_target_calibration_linear_svm_k{k}"].runner == "protocol3_source_plus_target"
        for alignment_method in ("procrustes", "hyperalignment", "mcca"):
            spec = registry[f"target_calibrated_{alignment_method}_k{k}"]
            assert spec.runner == "protocol3_target_calibrated_alignment"
            assert spec.protocol_category == 3
            assert spec.method_family == "target_calibrated_alignment"
        gaussian = registry[f"target_calibrated_gaussian_k{k}"]
        assert gaussian.runner == "protocol3_target_calibrated_gaussian"
        assert gaussian.method_family == "generative_augmentation"
        lora = registry[f"semi_supervised_lora_few_shot_k{k}"]
        assert lora.runner == "protocol3_lora_few_shot"
        assert lora.requires_torch is True


def test_protocol3_registry_audit_marks_wired_k_methods_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(all_protocols, "_module_available", lambda module: True)
    monkeypatch.setattr(all_protocols, "_torch_available", lambda: False)

    expected_available = set()
    for k in (1, 2, 4, 8, 16):
        expected_available.add(f"source_plus_target_calibration_logistic_k{k}")
        expected_available.add(f"source_plus_target_calibration_linear_svm_k{k}")
        expected_available.add(f"few_shot_target_calibrated_decoder_k{k}")
        for alignment_method in ("procrustes", "hyperalignment", "mcca"):
            expected_available.add(f"target_calibrated_{alignment_method}_k{k}")
        expected_available.add(f"target_calibrated_gaussian_k{k}")

    audit, audit_csv, strict_failures = build_registry_audit(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        protocols="3",
        strict_available=True,
    )

    assert audit_csv.exists()
    assert strict_failures == []
    protocol3 = audit.loc[audit["protocol_category"] == 3]
    available_protocol3 = protocol3.loc[protocol3["implementation_status"] == "available"]
    assert len(available_protocol3) > 0
    assert set(available_protocol3["method"]).issuperset(expected_available)
    assert set(available_protocol3["protocol_category"]) == {3}
    assert protocol3["uses_target_labels_for_fitting"].map(bool).all()

    by_method = audit.set_index("method")
    for method in expected_available:
        row = by_method.loc[method]
        assert row["implementation_status"] == "available"
        assert bool(row["inventory_only"]) is False
        assert int(row["protocol_category"]) == 3
        assert bool(row["uses_target_labels_for_fitting"]) is True

    for method in (
        "semi_supervised_lora_few_shot_k1",
        "semi_supervised_lora_few_shot_k2",
        "semi_supervised_lora_few_shot_k4",
        "semi_supervised_lora_few_shot_k8",
        "semi_supervised_lora_few_shot_k16",
        "target_calibrated_gan",
        "target_calibrated_diffusion",
    ):
        assert by_method.loc[method, "implementation_status"] == "skipped"


def test_source_plus_target_calibration_fit_predict_uses_calibration_rows() -> None:
    candidate = SimpleNamespace(
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
    )
    source_labels = np.asarray([0, 1, 2, 0, 1, 2])
    source_features = np.eye(6, 6, dtype=float)[:, :3]
    calibration_labels = np.asarray([0, 1, 2])
    calibration_features = np.eye(3, dtype=float)
    evaluation_features = np.eye(3, dtype=float)

    output = all_protocols._source_plus_target_calibrated_fit_predict(
        source_features=source_features,
        source_labels=source_labels,
        target_calibration_features=calibration_features,
        target_calibration_labels=calibration_labels,
        target_evaluation_features=evaluation_features,
        classes=np.asarray([0, 1, 2]),
        method_spec=method_registry()["source_plus_target_calibration_logistic_k1"],
        k_per_class=1,
        candidate=candidate,
        decoding={"max_iter": 500},
        protocol3={},
    )

    assert output["probabilities"].shape == (3, 3)
    assert np.allclose(output["probabilities"].sum(axis=1), 1.0)
    assert output["metadata"]["source_plus_target_calibration"] is True


def test_target_calibrated_alignment_receives_calibration_not_evaluation_labels(monkeypatch) -> None:
    import neureptrace.decoding.source_alignment as source_alignment

    seen = {}

    def fake_align_train_test_features(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            train_features=np.asarray(kwargs["train_features"]),
            test_features=np.asarray(kwargs["test_features"]),
            metadata={"alignment_target_calibrated": True},
        )

    monkeypatch.setattr(source_alignment, "align_train_test_features", fake_align_train_test_features)
    candidate = SimpleNamespace(decoder="logistic", emission_mode="uncalibrated", feature_preprocessor="none", pca_components=None, classifier_param=1.0)
    source_labels = np.asarray([0, 1, 2, 0, 1, 2])
    output = all_protocols._target_calibrated_alignment_fit_predict(
        source_features=np.eye(6, 6, dtype=float)[:, :3],
        source_labels=source_labels,
        source_subject_ids=np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"]),
        target_calibration_features=np.eye(3, dtype=float),
        target_calibration_labels=np.asarray([0, 1, 2]),
        target_evaluation_features=np.eye(3, dtype=float),
        classes=np.asarray([0, 1, 2]),
        method_spec=method_registry()["target_calibrated_procrustes_k1"],
        k_per_class=1,
        candidate=candidate,
        decoding={"max_iter": 500},
        protocol3={"alignment_method": "procrustes"},
    )

    assert "target_calibration_features" in seen
    assert "target_calibration_labels" in seen
    assert "target_labels" not in seen or seen["target_labels"] is None
    assert output["metadata"]["alignment_target_calibrated"] is True


def test_target_calibrated_gaussian_rejects_scored_target_labels_and_marks_synthetic_rows() -> None:
    from neureptrace.decoding.generative_augmentation import augment_training_features, generative_augmentation_config

    config = generative_augmentation_config(method="target_calibrated_gaussian", synthetic_per_class=2, random_state=13)
    with pytest.raises(ValueError, match="never accepts scored target_labels"):
        augment_training_features(
            np.eye(3),
            np.asarray([0, 1, 2]),
            config=config,
            target_labels=np.asarray([0, 1, 2]),
            target_calibration_features=np.eye(3),
            target_calibration_labels=np.asarray([0, 1, 2]),
        )

    candidate = SimpleNamespace(decoder="logistic", emission_mode="uncalibrated", feature_preprocessor="none", pca_components=None, classifier_param=1.0)
    output = all_protocols._target_calibrated_gaussian_fit_predict(
        source_features=np.eye(6, 6, dtype=float)[:, :3],
        source_labels=np.asarray([0, 1, 2, 0, 1, 2]),
        target_calibration_features=np.eye(3, dtype=float),
        target_calibration_labels=np.asarray([0, 1, 2]),
        target_evaluation_features=np.eye(3, dtype=float),
        classes=np.asarray([0, 1, 2]),
        method_spec=method_registry()["target_calibrated_gaussian_k1"],
        k_per_class=1,
        candidate=candidate,
        decoding={"max_iter": 500},
        protocol3={"synthetic_per_class": 2, "target_calibration_seed": 13},
    )
    assert output["metadata"]["augmentation_method"] == "target_calibrated_gaussian"
    assert output["metadata"]["synthetic_rows_marked"] is True
    assert output["probabilities"].shape == (3, 3)




def test_target_calibrated_gaussian_diagonal_fast_path_skips_matrix_power(monkeypatch) -> None:
    from neureptrace.decoding import generative_augmentation as gen

    def fail_matrix_power(*args, **kwargs):
        raise AssertionError("full covariance matrix power should not run for diagonal Gaussian")

    monkeypatch.setattr(gen, "_matrix_power", fail_matrix_power)
    features = np.vstack([np.eye(6, dtype=float), np.eye(6, dtype=float) + 0.1])
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    config = gen.generative_augmentation_config(
        method="target_calibrated_gaussian",
        synthetic_per_class=2,
        covariance_shrinkage=1.0,
        random_state=13,
    )

    output = gen.augment_training_features(
        features,
        labels,
        config=config,
        target_calibration_features=np.eye(3, 6, dtype=float),
        target_calibration_labels=np.asarray([0, 1, 2]),
    )

    assert output.n_synthetic == 6
    assert output.features.shape == (18, 6)


def test_lora_heavy_method_is_skipped_without_include_heavy(tmp_path) -> None:
    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="semi_supervised_lora_few_shot_k1",
        protocols="3",
        resume=False,
    )

    row = result.method_metadata.iloc[0]
    assert row["method"] == "semi_supervised_lora_few_shot_k1"
    assert row["status"] == "skipped"
    assert "include-heavy" in row["skip_reason"]


def test_lora_fit_predict_does_not_pass_evaluation_labels(monkeypatch) -> None:
    import neureptrace.decoding.semi_supervised_lora_few_shot as lora_module

    seen = {}

    def fake_fit_semi_supervised_lora_few_shot_decoder(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            probabilities=np.full((3, 3), 1.0 / 3.0),
            metadata={"semi_supervised_lora_uses_target_evaluation_labels": False},
        )

    monkeypatch.setattr(lora_module, "fit_semi_supervised_lora_few_shot_decoder", fake_fit_semi_supervised_lora_few_shot_decoder)
    output = all_protocols._semi_supervised_lora_few_shot_fit_predict(
        source_features=np.eye(6, 6, dtype=float)[:, :3],
        source_labels=np.asarray([0, 1, 2, 0, 1, 2]),
        source_subject_ids=np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"]),
        target_calibration_features=np.eye(3, dtype=float),
        target_calibration_labels=np.asarray([0, 1, 2]),
        target_evaluation_features=np.eye(3, dtype=float),
        classes=np.asarray([0, 1, 2]),
        method_spec=method_registry()["semi_supervised_lora_few_shot_k1"],
        k_per_class=1,
        candidate=SimpleNamespace(),
        decoding={},
        protocol3={"source_pretrain_epochs": 1, "target_adaptation_steps": 1, "hidden_dim": 8, "lora_rank": 2},
    )

    assert "target_evaluation_labels" not in seen
    assert seen["use_evaluation_features_unlabeled"] is True
    assert output["metadata"]["semi_supervised_lora_uses_target_evaluation_labels"] is False


def test_category3_calibration_evaluation_split_is_disjoint() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    calibration, evaluation = category3_calibration_evaluation_split(labels, calibration_per_class=1, seed=7)

    assert calibration.size == 3
    assert evaluation.size == 6
    assert np.intersect1d(calibration, evaluation).size == 0
    assert set(labels[calibration]) == {0, 1, 2}
    assert set(labels[evaluation]) == {0, 1, 2}
    validate_protocol_input_use(
        3,
        target_features_for_fitting=True,
        target_labels_for_fitting=True,
        calibration_indices=calibration,
        evaluation_indices=evaluation,
    )


def test_protocol3_rejects_overlapping_calibration_evaluation_rows() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_disjoint_calibration_evaluation([0, 1, 2], [2, 3, 4])

    with pytest.raises(ValueError, match="must be disjoint"):
        validate_protocol_input_use(
            3,
            target_features_for_fitting=True,
            target_labels_for_fitting=True,
            calibration_indices=[0, 1, 2],
            evaluation_indices=[2, 3, 4],
        )


def test_protocol4_only_runs_when_include_oracle_is_supplied() -> None:
    with pytest.raises(ValueError, match="requires --include-oracle"):
        _selected_methods(
            all_protocols={},
            methods="oracle_mcca_class_mean",
            protocols="4",
            include_oracle=False,
        )

    selected = _selected_methods(
        all_protocols={},
        methods="oracle_target_calibrated_mcca",
        protocols="4",
        include_oracle=True,
    )
    assert [spec.method for spec in selected] == ["oracle_target_calibrated_mcca"]


def test_all_selection_excludes_protocol4_without_oracle_flag() -> None:
    selected = _selected_methods(all_protocols={}, methods="all", protocols=None, include_oracle=False)
    assert selected
    assert all(spec.protocol_category != 4 for spec in selected)


def test_non_oracle_excludes_protocol4_even_when_oracle_is_allowed() -> None:
    selected = _selected_methods(
        all_protocols={},
        methods="source_loso_logistic,oracle_target_calibrated_mcca",
        protocols="1,4",
        include_oracle=True,
        non_oracle=True,
    )

    assert [spec.method for spec in selected] == ["source_loso_logistic"]


def test_available_only_excludes_inventory_only_methods(tmp_path, monkeypatch) -> None:
    def fake_source_loso(config_path, *, summary_path, inner_path, predictions_path, max_folds, progress_callback=None):
        summary = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "balanced_accuracy": 0.5,
                    "accuracy": 0.5,
                    "top2_accuracy": 0.75,
                    "top3_accuracy": 1.0,
                    "log_loss": 1.0,
                    "brier": 0.2,
                    "ece": 0.0,
                    "n_test_trials": 2,
                    "n_classes": 2,
                    "class_names": "0|1",
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
                    "prob_class_0": 0.8,
                    "prob_class_1": 0.2,
                }
            ]
        )
        summary.to_csv(summary_path, index=False)
        predictions.to_csv(predictions_path, index=False)
        pd.DataFrame().to_csv(inner_path, index=False)
        return summary

    monkeypatch.setattr(all_protocols, "_run_source_loso_method", fake_source_loso)

    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="source_loso_logistic,foundation_frozen_linear_probe",
        protocols="1",
        available_only=True,
        resume=False,
    )

    assert result.method_metadata["method"].tolist() == ["source_loso_logistic"]
    assert result.summary["method"].tolist() == ["source_loso_logistic"]
    assert not (tmp_path / "methods" / "foundation_frozen_linear_probe").exists()


def test_canary_cli_applies_tiny_available_source_loso_preset(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_source_loso(config_path, *, summary_path, inner_path, predictions_path, max_folds, progress_callback=None):
        config = load_config(config_path)
        seen["participants"] = config["participants"]["ids"]
        seen["window_centers"] = config["preprocessing"]["window_centers"]
        seen["max_folds"] = max_folds
        summary = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "balanced_accuracy": 0.5,
                    "accuracy": 0.5,
                    "top2_accuracy": 0.75,
                    "top3_accuracy": 1.0,
                    "log_loss": 1.0,
                    "brier": 0.2,
                    "ece": 0.0,
                    "n_test_trials": 2,
                    "n_classes": 2,
                    "class_names": "0|1",
                }
            ]
        )
        summary.to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(predictions_path, index=False)
        pd.DataFrame().to_csv(inner_path, index=False)
        return summary

    monkeypatch.setattr(all_protocols, "_run_source_loso_method", fake_source_loso)

    exit_code = main(["--config", "configs/bush_meg/all_protocols.yml", "--out-dir", str(tmp_path), "--canary", "--no-resume"])

    assert exit_code == 0
    assert seen["participants"] == "1,2,3"
    assert seen["max_folds"] == 1
    assert len(seen["window_centers"]) == 1
    metadata = pd.read_csv(tmp_path / "method_metadata.csv")
    assert metadata["method"].tolist() == ["source_loso_logistic"]


def test_missing_optional_family_is_written_as_skipped_metadata(tmp_path) -> None:
    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="foundation_frozen_linear_probe",
        protocols="1",
        resume=False,
    )
    row = result.method_metadata.iloc[0]
    assert row["method"] == "foundation_frozen_linear_probe"
    assert row["status"] == "skipped"
    assert "missing required config value" in row["skip_reason"]
    assert row["protocol_category"] == 1
    assert bool(row["uses_target_data"]) is False
    assert bool(row["uses_target_labels_for_fitting"]) is False
    assert bool(row["calibration_rows_disjoint_from_evaluation"]) is True
    assert bool(row["valid_for_strict_source_only"]) is True
    assert bool(row["valid_for_zero_calibration"]) is True
    assert bool(row["debug_upper_bound"]) is False
    assert result.summary.empty
    method_dir = tmp_path / "methods" / "foundation_frozen_linear_probe"
    status = json.loads((method_dir / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "method_skipped"
    assert status["method"] == "foundation_frozen_linear_probe"
    assert (method_dir / "run.log").exists()
    assert (method_dir / "summary.partial.csv").exists()
    assert (method_dir / "predictions.partial.csv").exists()
    log_text = (method_dir / "run.log").read_text(encoding="utf-8")
    assert '"stage": "configured"' in log_text
    assert '"stage": "checking_requirements"' in log_text
    assert '"stage": "method_skipped"' in log_text
    metadata_csv = tmp_path / "method_metadata.csv"
    assert metadata_csv.exists()
    assert "method_skipped" in metadata_csv.read_text(encoding="utf-8")


def test_tiny_run_limits_are_applied_to_method_config_before_execution(tmp_path) -> None:
    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="foundation_frozen_linear_probe",
        protocols="1",
        participants="1-6",
        smoke_participants="2,4,5",
        participant_limit=2,
        max_folds=9,
        fold_limit=1,
        window_limit=2,
        resume=False,
    )

    method_config = load_config(tmp_path / "methods" / "foundation_frozen_linear_probe" / "config.yml")
    assert method_config["participants"]["ids"] == "2,4"
    assert method_config["preprocessing"]["window_centers"] == [0.088, 0.136]
    assert method_config["source_loso"]["candidate_grid"]["window_sets"][0]["centers"] == [0.088, 0.136]

    row = result.method_metadata.iloc[0]
    assert row["status"] == "skipped"
    assert int(row["n_configured_participants"]) == 2

    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["participants"] == "1-6"
    assert provenance["effective_participants"] == "2,4"
    assert provenance["requested_max_folds"] == 9
    assert provenance["fold_limit"] == 1
    assert provenance["max_folds"] == 1
    assert provenance["window_limit"] == 2


def test_participant_limit_reduces_subjects_before_loading(tmp_path, monkeypatch) -> None:
    import neureptrace.bushmeg_all_protocols as all_protocols

    seen: dict[str, object] = {}
    method_dir = tmp_path / "methods" / "source_loso_logistic"
    method_dir.mkdir(parents=True)
    (method_dir / "run.log").write_text("STALE_LOG\n", encoding="utf-8")
    (method_dir / "summary.partial.csv").write_text("stale\n", encoding="utf-8")

    def fake_source_loso(config_path, *, summary_path, inner_path, predictions_path, max_folds, progress_callback=None):
        config = load_config(config_path)
        seen["participants"] = config["participants"]["ids"]
        seen["max_folds"] = max_folds
        seen["status_exists_before_loading"] = (summary_path.parent / "status.json").exists()
        seen["summary_path"] = summary_path
        return pd.DataFrame()

    monkeypatch.setattr(all_protocols, "_run_source_loso_method", fake_source_loso)

    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="source_loso_logistic",
        protocols="1",
        participants="1-6",
        participant_limit=2,
        fold_limit=1,
        resume=False,
    )

    assert seen["participants"] == "1,2"
    assert seen["max_folds"] == 1
    assert seen["status_exists_before_loading"] is True
    assert "STALE_LOG" not in (method_dir / "run.log").read_text(encoding="utf-8")
    assert result.method_metadata.loc[0, "status"] == "runnable"


def test_empty_csv_resume_reads_as_empty_dataframe(tmp_path) -> None:
    empty = tmp_path / "empty_predictions.csv"
    empty.write_text("\n", encoding="utf-8")

    frame = all_protocols._read_csv_if_nonempty(empty)

    assert frame.empty


def test_covariance_all_protocol_config_uses_single_candidate_grid(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_covariance(config_path, *, summary_path, inner_path, predictions_path, max_folds, progress_callback=None):
        config = load_config(config_path)
        seen["config"] = config
        summary = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "balanced_accuracy": 0.5,
                    "accuracy": 0.5,
                    "top2_accuracy": 1.0,
                    "top3_accuracy": 1.0,
                    "log_loss": 1.0,
                    "brier": 0.5,
                    "ece": 0.0,
                    "n_test_trials": 2,
                    "n_classes": 2,
                    "class_names": "0|1",
                }
            ]
        )
        summary.to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(predictions_path, index=False)
        pd.DataFrame().to_csv(inner_path, index=False)
        return summary

    monkeypatch.setattr(all_protocols, "_run_covariance_method", fake_covariance)

    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="covariance_loso",
        protocols="1",
        data_dir=tmp_path,
        participants="1,2,3",
        fold_limit=1,
        resume=False,
    )

    assert not result.summary.empty
    covariance_config = seen["config"]["covariance_loso"]
    grid = covariance_config["candidate_grid"]
    assert covariance_config["skip_inner_selection_when_single_candidate"] is True
    assert grid["c_grid"] == [1.0]
    assert grid["pca_components"] == [96]
    assert grid["feature_modes"] == ["logeuclidean_covariance"]


def test_rebuild_top_level_outputs_from_method_partials(tmp_path) -> None:
    spec = method_registry()["source_loso_logistic"]
    method_dir = tmp_path / "methods" / spec.method
    method_dir.mkdir(parents=True)
    config = load_config("configs/bush_meg/all_protocols.yml")
    (method_dir / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"methods": [spec.method]}) + "\n", encoding="utf-8")
    (method_dir / "status.json").write_text(
        json.dumps(
            {
                "method": spec.method,
                "stage": "fold_done",
                "updated_at_utc": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (method_dir / "summary.partial.csv").write_text(
        "\n".join(
            [
                "outer_test_subject,balanced_accuracy,accuracy,top2_accuracy,top3_accuracy,log_loss,brier,ece,n_train_subjects,n_test_trials,n_classes",
                "subj-1,0.5,0.6,0.7,0.8,1.2,0.3,0.1,2,10,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (method_dir / "predictions.partial.csv").write_text(
        "\n".join(
            [
                "outer_test_subject,trial_index,true_label,predicted_label,prob_class_0,prob_class_1,prob_class_2",
                "subj-1,0,0,0,0.8,0.1,0.1",
                "subj-1,1,1,1,0.2,0.5,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rebuilt = rebuild_all_protocol_outputs_from_partials(tmp_path, method_specs=[spec])

    assert rebuilt.summary_csv.exists()
    assert rebuilt.predictions_csv.exists()
    assert rebuilt.method_metadata_csv.exists()
    assert rebuilt.summary.loc[0, "method"] == spec.method
    assert rebuilt.summary.loc[0, "outer_test_subject"] == "subj-1"
    assert rebuilt.predictions.shape[0] == 2
    assert rebuilt.method_metadata.loc[0, "current_stage"] == "fold_done"
    assert rebuilt.method_metadata.loc[0, "status"] == "running"


def test_method_progress_records_fold_timeout(tmp_path) -> None:
    progress = MethodProgress(tmp_path / "methods" / "slow_method", method="slow_method", fold_timeout_seconds=0.001)
    progress.update("configured")
    progress.update("fold_start", outer_test_subject="subj-1", fold_index=1, n_folds=1)

    with pytest.raises(RunTimeoutError) as exc_info:
        time.sleep(0.02)
        progress.update("feature_start", outer_test_subject="subj-1", fold_index=1)

    assert exc_info.value.kind == "fold"
    assert exc_info.value.seconds == pytest.approx(0.001)
    progress.update(
        "method_failed",
        error_type=type(exc_info.value).__name__,
        error=str(exc_info.value),
        timeout_kind=exc_info.value.kind,
        timeout_seconds=exc_info.value.seconds,
    )
    status = json.loads((tmp_path / "methods" / "slow_method" / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "method_failed"
    assert status["timeout_kind"] == "fold"
    assert status["timeout_seconds"] == pytest.approx(0.001)


def test_fake_long_method_writes_status_before_interrupted(tmp_path, monkeypatch) -> None:
    import neureptrace.bushmeg_all_protocols as all_protocols

    def fake_long_source_loso(config_path, *, summary_path, inner_path, predictions_path, max_folds, progress_callback=None):
        status_path = summary_path.parent / "status.json"
        assert status_path.exists()
        before = json.loads(status_path.read_text(encoding="utf-8"))
        assert before["stage"] == "loading_subjects"
        time.sleep(0.05)
        if progress_callback is not None:
            progress_callback("feature_start", outer_test_subject="subj-1", fold_index=1)
        return pd.DataFrame()

    monkeypatch.setattr(all_protocols, "_run_source_loso_method", fake_long_source_loso)

    result = run_bushmeg_all_protocols(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods="source_loso_logistic",
        protocols="1",
        participant_limit=2,
        method_timeout_seconds=0.01,
        resume=False,
    )

    row = result.method_metadata.iloc[0]
    assert row["method"] == "source_loso_logistic"
    assert row["status"] == "failed"
    assert row["timeout_kind"] == "method"
    assert "method timeout exceeded" in row["blocked_reason"]
    method_dir = tmp_path / "methods" / "source_loso_logistic"
    status = json.loads((method_dir / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "method_failed"
    assert status["timeout_kind"] == "method"
    log_text = (method_dir / "run.log").read_text(encoding="utf-8")
    assert '"stage": "configured"' in log_text
    assert '"stage": "loading_subjects"' in log_text
    assert '"stage": "method_failed"' in log_text


def test_interrupted_partial_outputs_can_be_aggregated(tmp_path) -> None:
    spec = method_registry()["source_loso_logistic"]
    method_dir = tmp_path / "methods" / spec.method
    method_dir.mkdir(parents=True)
    config = load_config("configs/bush_meg/all_protocols.yml")
    (method_dir / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"methods": [spec.method]}) + "\n", encoding="utf-8")
    (method_dir / "status.json").write_text(
        json.dumps(
            {
                "method": spec.method,
                "stage": "method_failed",
                "updated_at_utc": "2026-01-01T00:00:00+00:00",
                "error": "fold timeout exceeded 1 seconds",
                "timeout_kind": "fold",
                "timeout_seconds": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (method_dir / "summary.partial.csv").write_text(
        "\n".join(
            [
                "outer_test_subject,balanced_accuracy,accuracy,top2_accuracy,top3_accuracy,log_loss,brier,ece,n_train_subjects,n_test_trials,n_classes",
                "subj-1,0.5,0.6,0.7,0.8,1.2,0.3,0.1,2,10,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (method_dir / "predictions.partial.csv").write_text(
        "\n".join(
            [
                "outer_test_subject,trial_index,true_label,predicted_label,prob_class_0,prob_class_1,prob_class_2",
                "subj-1,0,0,0,0.8,0.1,0.1",
                "subj-1,1,1,1,0.2,0.6,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rebuilt = rebuild_all_protocol_outputs_from_partials(tmp_path, method_specs=[spec])

    assert rebuilt.summary.loc[0, "outer_test_subject"] == "subj-1"
    assert rebuilt.predictions.shape[0] == 2
    assert rebuilt.method_metadata.loc[0, "status"] == "failed"
    assert rebuilt.method_metadata.loc[0, "timeout_kind"] == "fold"
    assert rebuilt.method_metadata.loc[0, "timeout_seconds"] == pytest.approx(1.0)


def test_partial_aggregation_coerces_subject_ids_before_metric_merge(tmp_path) -> None:
    spec = method_registry()["source_loso_logistic"]
    method_dir = tmp_path / "methods" / spec.method
    method_dir.mkdir(parents=True)
    (method_dir / "config.yml").write_text("participants:\n  ids: [1]\n", encoding="utf-8")
    (method_dir / "status.json").write_text(json.dumps({"stage": "fold_done", "method": spec.method}), encoding="utf-8")
    (method_dir / "summary.partial.csv").write_text(
        "outer_test_subject,n_test_trials,n_classes,class_names\n1,2,2,0|1\n",
        encoding="utf-8",
    )
    (method_dir / "predictions.partial.csv").write_text(
        "\n".join(
            [
                "outer_test_subject,trial_index,true_label,predicted_label,prob_class_0,prob_class_1",
                "1,0,0,0,0.8,0.2",
                "1,1,1,1,0.1,0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rebuilt = rebuild_all_protocol_outputs_from_partials(tmp_path, method_specs=[spec])

    assert rebuilt.summary.loc[0, "outer_test_subject"] == "1"
    assert rebuilt.summary.loc[0, "accuracy"] == pytest.approx(1.0)
    assert rebuilt.summary.loc[0, "balanced_accuracy"] == pytest.approx(1.0)


def test_registry_audit_writes_checkout_completeness_csv(tmp_path, monkeypatch) -> None:
    missing_method = _install_synthetic_missing_method(monkeypatch)
    audit, audit_csv, strict_failures = build_registry_audit(
        config_path="configs/bush_meg/all_protocols.yml",
        out_dir=tmp_path,
        methods=f"source_loso_logistic,{missing_method}",
        protocols="1",
        strict_available=True,
    )

    assert audit_csv == tmp_path / "registry_audit.csv"
    assert audit_csv.exists()
    required_columns = {
        "method",
        "protocol_category",
        "required_modules",
        "required_config_any",
        "requires_torch",
        "inventory_only",
        "implementation_status",
        "skip_reason",
    }
    assert required_columns.issubset(audit.columns)
    source_row = audit.loc[audit["method"] == "source_loso_logistic"].iloc[0]
    assert source_row["implementation_status"] == "available"
    assert source_row["inventory_only"] is False or str(source_row["inventory_only"]).lower() == "false"
    missing_row = audit.loc[audit["method"] == missing_method].iloc[0]
    assert missing_row["implementation_status"] == "skipped"
    assert missing_row["inventory_only"] is True or str(missing_row["inventory_only"]).lower() == "true"
    assert "neureptrace.decoding.definitely_missing_registry_audit_test" in missing_row["missing_required_modules"]
    audit_from_disk = pd.read_csv(audit_csv)
    disk_missing_row = audit_from_disk.loc[audit_from_disk["method"] == missing_method].iloc[0]
    assert disk_missing_row["implementation_status"] == "skipped"
    assert str(disk_missing_row["inventory_only"]).lower() == "true"
    assert "neureptrace.decoding.definitely_missing_registry_audit_test" in disk_missing_row["missing_required_modules"]
    assert strict_failures
    assert any(missing_method in failure for failure in strict_failures)


def test_strict_available_cli_fails_for_requested_missing_modules(tmp_path, monkeypatch) -> None:
    missing_method = _install_synthetic_missing_method(monkeypatch)
    exit_code = main(
        [
            "--config",
            "configs/bush_meg/all_protocols.yml",
            "--out-dir",
            str(tmp_path),
            "--methods",
            missing_method,
            "--protocols",
            "1",
            "--audit-registry",
            "--strict-available",
        ]
    )

    assert exit_code == 2
    audit_csv = tmp_path / "registry_audit.csv"
    assert audit_csv.exists()
    audit = pd.read_csv(audit_csv)
    row = audit.loc[audit["method"] == missing_method].iloc[0]
    assert row["requested"] is True or str(row["requested"]).lower() == "true"
    assert "neureptrace.decoding.definitely_missing_registry_audit_test" in row["missing_required_modules"]
