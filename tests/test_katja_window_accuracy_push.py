from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil

import neureptrace.katja_window_accuracy_push as accuracy_push_module
import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.katja_window_structure import (
    TargetCalibrationPartition,
    balanced_window_sampling_weights,
    causal_trial_decode,
    combine_hierarchical_probabilities,
    compose_template_order_finger_probabilities,
    conditional_finger_targets,
    ensemble_prediction_bundles,
    estimate_state_duration_priors,
    estimate_state_stay_probabilities,
    explicit_duration_trial_decode,
    learn_finger_templates,
    match_probability_marginals,
    structured_trial_decode,
    write_prediction_bundle,
)
from neureptrace.katja_julia_window_benchmark import select_nested_trial_splits
from neureptrace.katja_window_accuracy_push import (
    CLASSIFICATION_INFORMATION_BOUNDARY,
    DIRECT_JULIA_COMPARISON_METHODS,
    INDEPENDENT_DESIGN_PROVENANCE,
    _common_curve_cohort_summaries,
    _metric_and_append,
    _method_comparison_scope,
    _paired_push_statistics,
    _screen_adapter_for_outer_target,
    _validate_full_prediction_bundles,
    aggregate_accuracy_push_shards,
    apply_probability_temperature,
    fit_probability_temperature,
    run_accuracy_push,
    select_window_adaptation_hyperparameters,
    validate_baseline_reproduction,
)


def test_hierarchical_probabilities_sum_to_one_and_soft_targets_match_hard_finger() -> None:
    ratios = np.asarray(
        [
            [1.0, 0, 0, 0, 0, 0],
            [0.2, 0.7, 0.1, 0, 0, 0],
            [0.4, 0, 0, 0.6, 0, 0],
        ],
        dtype=float,
    )
    hard = np.asarray([0, 1, 3])
    targets, active = conditional_finger_targets(ratios, hard)
    assert active.tolist() == [False, True, True]
    assert np.argmax(targets[active], axis=1).tolist() == [0, 2]
    press = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    combined = combine_hierarchical_probabilities(press, targets + 1e-3)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(combined[:, 0], press[:, 0], atol=1e-12)


def test_transductive_marginal_matching_uses_probabilities_and_calibration_prior() -> None:
    rng = np.random.default_rng(17)
    probabilities = rng.dirichlet(np.ones(6), size=2000)
    prior = np.asarray([0.28, 0.14, 0.13, 0.16, 0.15, 0.14])
    adjusted, biases = match_probability_marginals(probabilities, prior)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(adjusted.mean(axis=0), prior, atol=2e-8)
    assert biases.shape == (6,)


def test_template_order_composition_maps_positions_to_calibration_fingers() -> None:
    templates = ((3, 4, 2, 5, 1), (3, 1, 5, 2, 4))
    press = np.asarray([[0.1, 0.9], [0.2, 0.8]])
    order = np.zeros((2, 6), dtype=float)
    order[0, 2] = 1.0
    order[1, 4] = 1.0
    template = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    composed = compose_template_order_finger_probabilities(
        press, order, template, templates
    )
    np.testing.assert_allclose(composed.sum(axis=1), 1.0)
    assert np.argmax(composed, axis=1).tolist() == [4, 2]


def test_classification_information_boundary_excludes_evaluation_annotations() -> None:
    boundary = CLASSIFICATION_INFORMATION_BOUNDARY
    assert boundary["evaluation_inputs"] == [
        "Julia-supplied cached MEG window samples",
        "held-out participant identity for selecting the target adapter",
        "trial membership",
        "chronological cache-row order within each trial",
    ]
    forbidden = " ".join(boundary["forbidden_evaluation_inputs"])
    for annotation in (
        "finger labels",
        "sequence labels",
        "press-order labels",
        "press ratios",
        "timestamps",
    ):
        assert annotation in forbidden
    assert "does not use evaluation annotations" in boundary["structured_inference"]


def test_only_independent_window_methods_are_direct_julia_comparisons() -> None:
    methods = [
        *DIRECT_JULIA_COMPARISON_METHODS,
        "trial_transformer_ensemble",
        "hierarchical_tcn_ensemble_structured",
        "hybrid_ensemble_structured",
    ]
    scope = _method_comparison_scope(methods).set_index("method")
    assert scope.loc[list(DIRECT_JULIA_COMPARISON_METHODS), "direct_julia_comparison"].all()
    assert not scope.loc[
        [
            "trial_transformer_ensemble",
            "hierarchical_tcn_ensemble_structured",
            "hybrid_ensemble_structured",
        ],
        "direct_julia_comparison",
    ].any()
    assert scope.loc[
        "trial_transformer_ensemble", "comparison_scope"
    ] == "supplementary_trial_context"
    assert scope.loc[
        "hierarchical_tcn_ensemble_structured", "comparison_scope"
    ] == "supplementary_task_structure"


def test_method_design_provenance_distinguishes_endpoint_from_model() -> None:
    provenance = INDEPENDENT_DESIGN_PROVENANCE
    assert provenance["collaborator_model_architecture_received"] is False
    assert provenance["collaborator_training_or_adaptation_code_received"] is False
    assert provenance["collaborator_split_function_received"] is False
    assert provenance["copied_collaborator_model_components"] is False
    assert "minimum-overlap relabel rule" in " ".join(
        provenance["shared_external_artifacts"]
    )
    assert "independent NeuRepTrace implementations" in provenance["scope"]


def test_temperature_scaling_is_fit_only_from_supplied_calibration_rows() -> None:
    probabilities = np.asarray(
        [
            [0.45, 0.35, 0.05, 0.05, 0.05, 0.05],
            [0.35, 0.45, 0.05, 0.05, 0.05, 0.05],
            [0.45, 0.35, 0.05, 0.05, 0.05, 0.05],
            [0.35, 0.45, 0.05, 0.05, 0.05, 0.05],
        ]
    )
    labels = np.asarray([0, 1, 0, 1])
    temperature = fit_probability_temperature(probabilities, labels)
    calibrated = apply_probability_temperature(probabilities, temperature)
    assert temperature < 1.0
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)
    before = -np.log(probabilities[np.arange(labels.size), labels]).mean()
    after = -np.log(calibrated[np.arange(labels.size), labels]).mean()
    assert after < before


def test_fixed_max_complement_uses_identical_evaluation_rows_across_k() -> None:
    sequence = np.repeat(np.arange(4), 8)
    trial = np.arange(sequence.size)
    splits = select_nested_trial_splits(
        sequence,
        trial,
        k_values=(1, 2, 4),
        seed=3,
        context="s05",
        split_mode="fixed_max_complement",
    )
    reference = splits[1].evaluation_rows
    assert all(np.array_equal(split.evaluation_rows, reference) for split in splits.values())
    assert splits[1].reserved_rows.size > 0
    assert not np.intersect1d(splits[1].calibration_rows, splits[1].evaluation_rows).size


def test_partition_rejects_any_target_row_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        TargetCalibrationPartition(
            calibration_indices=np.asarray([1, 2]),
            evaluation_indices=np.asarray([2, 3]),
            reserved_indices=np.asarray([4]),
            split_seed=0,
        )


def _template_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    template_a = (3, 4, 2, 5, 1)
    template_b = (3, 1, 5, 2, 4)
    labels: list[int] = []
    order: list[int] = []
    trials: list[int] = []
    calibration: list[int] = []
    evaluation: list[int] = []
    for trial, template in enumerate((template_a, template_b, template_a)):
        for rest_or_press in range(11):
            if rest_or_press % 2:
                position = (rest_or_press + 1) // 2
                label = template[position - 1]
            else:
                position = 0
                label = 0
            for _ in range(2):
                row = len(labels)
                labels.append(label)
                order.append(position)
                trials.append(trial)
                (calibration if trial < 2 else evaluation).append(row)
    return (
        np.asarray(labels),
        np.asarray(order),
        np.asarray(trials),
        np.asarray(calibration),
        np.asarray(evaluation),
    )


def test_templates_and_duration_priors_use_calibration_not_evaluation_labels() -> None:
    labels, order, trials, calibration, evaluation = _template_fixture()
    templates = learn_finger_templates(
        labels,
        order,
        trials,
        calibration_indices=calibration,
        evaluation_indices=evaluation,
    )
    assert set(templates) == {(3, 4, 2, 5, 1), (3, 1, 5, 2, 4)}
    stay = estimate_state_stay_probabilities(
        labels,
        order,
        trials,
        fitting_indices=calibration,
        evaluation_indices=evaluation,
    )
    perturbed = labels.copy()
    perturbed_order = order.copy()
    perturbed[evaluation] = np.roll(perturbed[evaluation], 1)
    perturbed_order[evaluation] = np.roll(perturbed_order[evaluation], 3)
    templates_perturbed = learn_finger_templates(
        perturbed,
        perturbed_order,
        trials,
        calibration_indices=calibration,
        evaluation_indices=evaluation,
    )
    stay_perturbed = estimate_state_stay_probabilities(
        perturbed,
        perturbed_order,
        trials,
        fitting_indices=calibration,
        evaluation_indices=evaluation,
    )
    assert templates_perturbed == templates
    np.testing.assert_array_equal(stay_perturbed, stay)


def test_explicit_duration_priors_ignore_evaluation_labels() -> None:
    labels, order, trials, calibration, evaluation = _template_fixture()
    source = calibration[trials[calibration] == 0]
    target_calibration = calibration[trials[calibration] == 1]
    priors = estimate_state_duration_priors(
        labels,
        order,
        trials,
        source_indices=source,
        calibration_indices=target_calibration,
        evaluation_indices=evaluation,
        calibration_prior_strength=2.0,
    )
    perturbed_labels = labels.copy()
    perturbed_order = order.copy()
    perturbed_labels[evaluation] = np.roll(perturbed_labels[evaluation], 3)
    perturbed_order[evaluation] = np.roll(perturbed_order[evaluation], 5)
    perturbed = estimate_state_duration_priors(
        perturbed_labels,
        perturbed_order,
        trials,
        source_indices=source,
        calibration_indices=target_calibration,
        evaluation_indices=evaluation,
        calibration_prior_strength=2.0,
    )
    np.testing.assert_array_equal(perturbed.means, priors.means)
    np.testing.assert_array_equal(perturbed.scales, priors.scales)
    np.testing.assert_array_equal(perturbed.minimums, priors.minimums)
    np.testing.assert_array_equal(perturbed.maximums, priors.maximums)

    with pytest.raises(ValueError, match="cannot use evaluation rows"):
        estimate_state_duration_priors(
            labels,
            order,
            trials,
            source_indices=np.concatenate((source, evaluation[:1])),
            calibration_indices=target_calibration,
            evaluation_indices=evaluation,
        )


def test_template_learning_rejects_a_third_calibration_template() -> None:
    labels, order, trials, calibration, evaluation = _template_fixture()
    third_labels = np.repeat([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0], 2)
    third_order = np.repeat([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0], 2)
    labels = np.concatenate((labels, third_labels))
    order = np.concatenate((order, third_order))
    trials = np.concatenate((trials, np.full(third_labels.size, 3)))
    calibration = np.concatenate(
        (calibration, np.arange(labels.size - third_labels.size, labels.size))
    )
    with pytest.raises(ValueError, match="exactly 2"):
        learn_finger_templates(
            labels,
            order,
            trials,
            calibration_indices=calibration,
            evaluation_indices=evaluation,
        )


def test_structured_decoder_emits_exactly_five_ordered_calibration_template_presses() -> None:
    template = (3, 4, 2, 5, 1)
    labels = np.repeat([0, 3, 0, 4, 0, 2, 0, 5, 0, 1, 0], 3)
    probabilities = np.full((labels.size, 6), 0.01)
    probabilities[np.arange(labels.size), labels] = 0.95
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    decoded = structured_trial_decode(probabilities, (template, (3, 1, 5, 2, 4)))
    assert decoded.template == template
    assert len(decoded.press_segments) == 5
    assert tuple(segment[2] for segment in decoded.press_segments) == template
    assert all(left[1] <= right[0] for left, right in zip(decoded.press_segments, decoded.press_segments[1:]))


def test_explicit_duration_decoder_emits_five_ordered_template_presses() -> None:
    template = (3, 4, 2, 5, 1)
    labels = np.repeat([0, 3, 0, 4, 0, 2, 0, 5, 0, 1, 0], 3)
    probabilities = np.full((labels.size, 6), 0.01)
    probabilities[np.arange(labels.size), labels] = 0.95
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    source_labels = np.tile(labels, 2)
    source_order = np.tile(np.repeat([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0], 3), 2)
    source_trials = np.repeat([0, 1], labels.size)
    priors = estimate_state_duration_priors(
        source_labels,
        source_order,
        source_trials,
        source_indices=np.arange(labels.size),
        calibration_indices=np.arange(labels.size, 2 * labels.size),
    )
    decoded = explicit_duration_trial_decode(
        probabilities,
        (template, (3, 1, 5, 2, 4)),
        duration_priors=priors,
    )
    assert decoded.template == template
    assert len(decoded.press_segments) == 5
    assert tuple(segment[2] for segment in decoded.press_segments) == template
    assert all(left[1] <= right[0] for left, right in zip(decoded.press_segments, decoded.press_segments[1:]))


def test_structured_decoder_uses_auxiliary_template_predictions() -> None:
    templates = ((3, 4, 2, 5, 1), (3, 1, 5, 2, 4))
    probabilities = np.full((33, 6), 1.0 / 6.0)
    template_probabilities = np.tile(np.asarray([0.01, 0.99]), (33, 1))
    decoded = structured_trial_decode(
        probabilities,
        templates,
        template_probabilities=template_probabilities,
    )
    assert decoded.template == templates[1]


def test_causal_decode_prefix_is_invariant_to_future_windows() -> None:
    rng = np.random.default_rng(9)
    probabilities = rng.dirichlet(np.ones(6), size=40)
    order_probabilities = rng.dirichlet(np.ones(6), size=40)
    overlap_probabilities = rng.uniform(size=40)
    template_probabilities = rng.dirichlet(np.ones(2), size=40)
    templates = ((3, 4, 2, 5, 1), (3, 1, 5, 2, 4))
    first = causal_trial_decode(
        probabilities,
        templates,
        order_probabilities=order_probabilities,
        overlap_probabilities=overlap_probabilities,
        template_probabilities=template_probabilities,
    )
    altered = probabilities.copy()
    altered[20:] = rng.dirichlet(np.ones(6), size=20)
    altered_order = order_probabilities.copy()
    altered_order[20:] = rng.dirichlet(np.ones(6), size=20)
    altered_overlap = overlap_probabilities.copy()
    altered_overlap[20:] = rng.uniform(size=20)
    altered_template = template_probabilities.copy()
    altered_template[20:] = rng.dirichlet(np.ones(2), size=20)
    second = causal_trial_decode(
        altered,
        templates,
        order_probabilities=altered_order,
        overlap_probabilities=altered_overlap,
        template_probabilities=altered_template,
    )
    np.testing.assert_array_equal(first[:20], second[:20])


def test_prediction_ensemble_requires_same_split_rows_and_separate_model_seeds(tmp_path) -> None:
    rows = np.asarray([8, 10, 12])
    first = np.eye(6, dtype=float)[[0, 1, 2]]
    second = np.eye(6, dtype=float)[[0, 2, 2]]
    paths = []
    for model_seed, probabilities in ((13, first), (29, second)):
        path = tmp_path / f"model_{model_seed}.npz"
        write_prediction_bundle(
            path,
            row_indices=rows,
            probabilities=probabilities,
            split_seed=7,
            model_seed=model_seed,
            method="m",
            auxiliary={"order_logits": np.zeros((3, 6))},
        )
        paths.append(path)
    ensemble = ensemble_prediction_bundles(paths)
    assert ensemble["split_seed"] == 7
    assert ensemble["model_seeds"] == (13, 29)
    np.testing.assert_array_equal(ensemble["row_indices"], rows)
    np.testing.assert_allclose(ensemble["probabilities"], (first + second) / 2)

    mismatch = tmp_path / "mismatch.npz"
    write_prediction_bundle(
        mismatch,
        row_indices=np.asarray([8, 10, 99]),
        probabilities=second,
        split_seed=7,
        model_seed=47,
        method="m",
    )
    with pytest.raises(ValueError, match="identical evaluation rows"):
        ensemble_prediction_bundles((paths[0], mismatch))


def test_balanced_sampling_equalizes_subject_trial_and_class_groups() -> None:
    subject = np.asarray([0, 0, 0, 1, 1, 1])
    trial = np.asarray([0, 0, 1, 0, 1, 1])
    labels = np.asarray([0, 0, 1, 0, 1, 1])
    weights = balanced_window_sampling_weights(subject, trial, labels)
    assert weights.shape == (6,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[2] > weights[0]


def test_adaptation_selection_never_predicts_or_fits_evaluation_rows() -> None:
    accesses: list[tuple[str, tuple[int, ...]]] = []

    class FakeModel:
        adapter_steps = 2
        last_block_steps = 2
        full_finetune_steps = 2

        def clone_source(self, *, random_state):
            return FakeModel()

        def register_target_calibration_labels(self, indices, **_labels):
            accesses.append(("fit", tuple(np.asarray(indices).tolist())))
            return self

        def adapt_target_indices(self, indices, **_kwargs):
            accesses.append(("adapt", tuple(np.asarray(indices).tolist())))
            return self

        def predict_proba_indices(self, indices):
            rows = np.asarray(indices)
            accesses.append(("predict", tuple(rows.tolist())))
            probabilities = np.full((rows.size, 6), 0.01)
            probabilities[:, 1] = 0.95
            return probabilities / probabilities.sum(axis=1, keepdims=True)

    sequence = np.repeat([0, 1], 6)
    trial = np.repeat(np.arange(6), 2)
    calibration = np.asarray([0, 1, 2, 3, 6, 7, 8, 9])
    evaluation = np.asarray([4, 5, 10, 11])
    labels = np.ones(12, dtype=int)
    candidates = (
        {"learning_rate": 0.001, "source_replay_weight": 0.1, "step_multiplier": 0.5},
        {"learning_rate": 0.002, "source_replay_weight": 0.1, "step_multiplier": 1.0},
    )
    select_window_adaptation_hyperparameters(
        FakeModel(),
        calibration_indices=calibration,
        evaluation_indices=evaluation,
        sequence_ids=sequence,
        trial_ids=trial,
        finger_labels=labels,
        raw_finger_labels=labels,
        order_labels=labels,
        overlap_targets=np.ones(12),
        press_ratios=np.eye(6)[labels],
        candidates=candidates,
        seed=4,
    )
    for _operation, rows in accesses:
        assert set(rows).isdisjoint(evaluation.tolist())


def test_small_k_adaptation_selection_uses_the_full_audit_schema() -> None:
    class FakeModel:
        adapter_steps = 2
        last_block_steps = 2
        full_finetune_steps = 2

    sequence = np.repeat([0, 1], 4)
    trial = np.arange(8)
    calibration = np.asarray([0, 4])
    evaluation = np.asarray([1, 2, 3, 5, 6, 7])
    labels = np.ones(8, dtype=int)
    candidate = {
        "learning_rate": 0.001,
        "source_replay_weight": 0.1,
        "step_multiplier": 0.5,
    }
    selected, rows = select_window_adaptation_hyperparameters(
        FakeModel(),
        calibration_indices=calibration,
        evaluation_indices=evaluation,
        sequence_ids=sequence,
        trial_ids=trial,
        finger_labels=labels,
        raw_finger_labels=labels,
        order_labels=labels,
        overlap_targets=np.ones(8),
        press_ratios=np.eye(6)[labels],
        candidates=(candidate,),
        seed=4,
    )
    assert selected == candidate
    assert set(rows[0]) == {
        *candidate,
        "candidate_index",
        "selection_status",
        "n_inner_training_trials",
        "n_inner_validation_trials",
        "inner_validation_accuracy",
        "evaluation_rows_accessed",
    }
    assert rows[0]["n_inner_training_trials"] == 2
    assert rows[0]["n_inner_validation_trials"] == 0
    assert rows[0]["evaluation_rows_accessed"] is False


def test_adapter_screen_excludes_the_outer_target_before_inner_loso(
    tmp_path, monkeypatch
) -> None:
    observed_domains: list[tuple[tuple[int, ...], int]] = []

    class Candidate:
        best_source_epoch_ = 1

        def register_target_calibration_labels(self, indices, **_labels):
            self.calibration = np.asarray(indices)
            return self

        def adapt_target_indices(self, indices, **_kwargs):
            np.testing.assert_array_equal(indices, self.calibration)
            return self

        def predict_proba_indices(self, indices):
            probabilities = np.zeros((len(indices), 6), dtype=float)
            probabilities[:, 0] = 1.0
            return probabilities

    def fake_fit(_args, **kwargs):
        rows = np.asarray(kwargs["source_indices"])
        domains = np.asarray(kwargs["subject_ids"])[rows]
        missing = set(range(10)) - {0} - set(np.unique(domains).tolist())
        assert len(missing) == 1
        pseudo_target = missing.pop()
        observed_domains.append((tuple(np.unique(domains).tolist()), pseudo_target))
        assert 0 not in domains
        return Candidate()

    monkeypatch.setattr(
        "neureptrace.katja_window_accuracy_push._fit_hierarchical_source_model",
        fake_fit,
    )
    subject_ids = np.repeat(np.arange(10), 16)
    n_rows = subject_ids.size
    sequence_ids = np.tile(np.repeat(np.arange(4), 4), 10)
    trial_ids = np.tile(np.arange(16), 10)
    moments = {
        "subjects": np.arange(10),
        "counts": np.full(10, 8),
        "sums": np.zeros((10, 2)),
        "squared_sums": np.full((10, 2), 8.0),
    }
    args = argparse.Namespace(
        resume=False,
        adapter_screen_fold_limit=2,
        adapter_screen_source_epochs=1,
        adapter_screen_k=1,
        adapter_configurations="low_rank:8,channel_affine_residual:16",
    )
    selected = _screen_adapter_for_outer_target(
        args,
        target="s05",
        output_dir=tmp_path,
        model_seed=0,
        window_store=np.zeros((n_rows, 2, 2), dtype=np.float32),
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        training_labels=np.zeros(n_rows, dtype=int),
        raw_finger_labels=np.zeros(n_rows, dtype=int),
        sequence_ids=sequence_ids,
        order_labels=np.zeros(n_rows, dtype=int),
        overlap_targets=np.zeros(n_rows),
        press_ratios=np.eye(6, dtype=np.float32)[np.zeros(n_rows, dtype=int)],
        moments=moments,
        moment_domains=np.arange(10),
        subject_means=np.zeros((10, 2), dtype=np.float32),
        subject_stds=np.ones((10, 2), dtype=np.float32),
    )
    assert selected["outer_test_subject"] == "s05"
    assert selected["outer_target_data_used"] is False
    assert selected["outer_target_labels_used"] is False
    assert len(observed_domains) == 4
    assert all(0 not in domains for domains, _pseudo_target in observed_domains)
    assert all(
        pseudo_target not in domains for domains, pseudo_target in observed_domains
    )


def test_exact_baseline_reproduction_gate(tmp_path) -> None:
    (tmp_path / "validation.json").write_text(
        json.dumps({"all_required_checks_pass": True}), encoding="utf-8"
    )
    with (tmp_path / "summary_subject_sem.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "k_trials_per_sequence",
                "n_subjects",
                "mean_accuracy_raw_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "method": "progressive_full",
                    "k_trials_per_sequence": 10,
                    "n_subjects": 10,
                    "mean_accuracy_raw_labels": 0.5335594321714971,
                },
                {
                    "method": "progressive_full",
                    "k_trials_per_sequence": 20,
                    "n_subjects": 9,
                    "mean_accuracy_raw_labels": 0.5602636869077369,
                },
            ]
        )
    result = validate_baseline_reproduction(tmp_path)
    assert result["validation_passed"] is True


def _write_baseline_gate(root) -> None:
    root.mkdir(parents=True)
    (root / "validation.json").write_text(
        json.dumps({"all_required_checks_pass": True}), encoding="utf-8"
    )
    with (root / "summary_subject_sem.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "k_trials_per_sequence", "n_subjects", "mean_accuracy_raw_labels"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"method": "progressive_full", "k_trials_per_sequence": 10, "n_subjects": 10, "mean_accuracy_raw_labels": 0.5335594321714971},
                {"method": "progressive_full", "k_trials_per_sequence": 20, "n_subjects": 9, "mean_accuracy_raw_labels": 0.5602636869077369},
            ]
        )


def _write_push_cache(path) -> None:
    rng = np.random.default_rng(17)
    templates = ((3, 4, 2, 5, 1), (3, 1, 5, 2, 4))
    windows = []
    finger = []
    overlap = []
    subject = []
    sequence = []
    trial = []
    order = []
    ratios = []
    for subject_index in range(10):
        trial_index = 0
        for sequence_index in range(4):
            template = templates[sequence_index % 2]
            for _repeat in range(2):
                states = [0]
                for value in template:
                    states.extend((value, 0))
                for state_index, label in enumerate(states):
                    sample = rng.normal(scale=0.15, size=(4, 3)).astype(np.float32)
                    sample[:, label % 3] += 0.8
                    windows.append(sample)
                    finger.append(label)
                    overlap.append(float(label > 0))
                    subject.append(subject_index)
                    sequence.append(sequence_index)
                    trial.append(trial_index)
                    press_position = (state_index + 1) // 2 if label > 0 else 0
                    order.append(press_position)
                    ratios.append(np.eye(6, dtype=np.float32)[label])
                trial_index += 1
    np.savez(
        path,
        meg_windows=np.asarray(windows, dtype=np.float32),
        finger_ids=np.asarray(finger, dtype=np.int64),
        press_overlap_fraction=np.asarray(overlap, dtype=np.float32),
        subject_indices=np.asarray(subject, dtype=np.int64),
        sequence_id=np.asarray(sequence, dtype=np.int64),
        trial_id=np.asarray(trial, dtype=np.int64),
        press_order=np.asarray(order, dtype=np.int64),
        press_ratios=np.asarray(ratios, dtype=np.float32),
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_tiny_accuracy_push_run_writes_member_ensemble_and_structured_rows(tmp_path) -> None:
    cache = tmp_path / "cache.npz"
    baseline = tmp_path / "baseline"
    output = tmp_path / "result"
    _write_push_cache(cache)
    _write_baseline_gate(baseline)
    args = argparse.Namespace(
        cache=str(cache),
        out_dir=str(output),
        baseline_results=str(baseline),
        raw_window_cache=None,
        targets="s05",
        k_values="1",
        split_seeds="0",
        model_seeds="0",
        context_modes="none",
        minimum_overlap=0.2,
        feature_batch_size=128,
        hidden_units=8,
        num_blocks=1,
        adapter_rank=8,
        adapter_kind="low_rank",
        screen_adapters=False,
        selected_adapter_config=None,
        screen_only=False,
        source_epochs=1,
        source_validation_patience=1,
        adapter_steps=1,
        last_block_steps=0,
        full_finetune_steps=0,
        tune_adaptation=False,
        adaptation_learning_rates="0.001",
        source_replay_weights="0.1",
        adaptation_step_multipliers="1.0",
        batch_size=64,
        sequence_loss_weight=0.15,
        order_loss_weight=0.3,
        overlap_loss_weight=0.3,
        context_hidden_units=8,
        context_heads=2,
        context_source_epochs=1,
        context_adaptation_steps=1,
        context_batch_trials=2,
        device="cpu",
        max_folds=1,
        resume=False,
    )
    run_accuracy_push(args)
    rows = list(csv.DictReader((output / "fold_results.csv").open(newline="", encoding="utf-8")))
    methods = {row["method"] for row in rows}
    assert methods == {
        "hierarchical_source_only",
        "hierarchical_tcn_single",
        "hierarchical_tcn_ensemble",
        "hierarchical_tcn_ensemble_structured",
        "hierarchical_tcn_ensemble_causal",
    }
    assert (output / "predictions" / "hierarchical_tcn" / "target=s05" / "split_seed=0" / "k=1" / "model_seed=0.npz").exists()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["all_required_checks_pass"] is True
    feasibility = json.loads((output / "feasibility.json").read_text(encoding="utf-8"))
    assert feasibility["s05"]["feasible_k_values"] == [1]
    assert feasibility["s05"]["fixed_evaluation_pool_k"] == 1
    assert (output / "headline_results.csv").exists()

    member_partial = output / "member_results.partial.csv"
    duplicated_members = pd.read_csv(member_partial)
    pd.concat((duplicated_members, duplicated_members.iloc[[0]]), ignore_index=True).to_csv(
        member_partial, index=False
    )
    args.resume = True
    run_accuracy_push(args)
    resumed_members = pd.read_csv(member_partial)
    identity = ["target", "split_seed", "k_trials_per_sequence", "method", "model_seed"]
    assert not resumed_members.duplicated(identity).any()

    second = tmp_path / "second_shard"
    shutil.copytree(output, second)
    for name in ("fold_results.csv", "member_results.csv"):
        frame = pd.read_csv(second / name)
        frame["target"] = "s06"
        frame["target_index"] = 1
        frame.to_csv(second / name, index=False)
    selected_path = second / "selected_adapter_configs.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["s06"] = selected.pop("s05")
    selected["s06"]["outer_test_subject"] = "s06"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    feasibility_path = second / "feasibility.json"
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    feasibility["s06"] = feasibility.pop("s05")
    feasibility_path.write_text(json.dumps(feasibility), encoding="utf-8")
    combined = tmp_path / "combined"
    aggregate_accuracy_push_shards(
        [output, second], output_dir=combined, baseline_results=baseline
    )
    combined_validation = json.loads(
        (combined / "validation.json").read_text(encoding="utf-8")
    )
    assert combined_validation["all_required_checks_pass"] is True
    assert combined_validation["n_shards"] == 2
    assert set(pd.read_csv(combined / "fold_results.csv")["target"]) == {"s05", "s06"}

    with pytest.raises(RuntimeError, match="Full accuracy-push design validation failed"):
        aggregate_accuracy_push_shards(
            [output, second],
            output_dir=tmp_path / "strict_combined",
            baseline_results=baseline,
            require_full_design=True,
        )


def test_full_bundle_validation_checks_rows_and_source_invariance(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(accuracy_push_module, "JULIA_SUBJECTS", ("s05",))
    monkeypatch.setattr(accuracy_push_module, "DEFAULT_SEEDS", (0,))
    monkeypatch.setattr(
        accuracy_push_module,
        "PREDICTION_FAMILIES",
        ("hierarchical_source_only",),
    )
    shard = tmp_path / "shard"
    feasibility = {"s05": {"feasible_k_values": [1, 3]}}
    probabilities = np.full((2, 6), 1.0 / 6.0, dtype=np.float32)

    def write(k: int, rows: np.ndarray, values: np.ndarray = probabilities) -> None:
        write_prediction_bundle(
            shard
            / "predictions"
            / "hierarchical_source_only"
            / "target=s05"
            / "split_seed=0"
            / f"k={k}"
            / "model_seed=0.npz",
            row_indices=rows,
            probabilities=values,
            split_seed=0,
            model_seed=0,
            method="hierarchical_source_only",
        )

    write(1, np.asarray([10, 11]))
    write(3, np.asarray([10, 12]))
    with pytest.raises(RuntimeError, match="Evaluation rows differ across k"):
        _validate_full_prediction_bundles({"s05": shard}, feasibility)

    changed = probabilities.copy()
    changed[0] = np.asarray([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    write(3, np.asarray([10, 11]), changed)
    with pytest.raises(RuntimeError, match="Source-only probabilities changed across k"):
        _validate_full_prediction_bundles({"s05": shard}, feasibility)

    write(3, np.asarray([10, 11]))
    validation = _validate_full_prediction_bundles({"s05": shard}, feasibility)
    assert validation["n_prediction_bundles_checked"] == 2
    assert validation["prediction_rows_identical_across_k_families_and_model_seeds"] is True


def test_offline_context_rows_are_flagged_as_using_future_windows(tmp_path) -> None:
    from types import SimpleNamespace

    split = SimpleNamespace(
        k=1,
        calibration_rows=np.asarray([0]),
        evaluation_rows=np.asarray([1, 2]),
        calibration_trials=np.asarray([0]),
        evaluation_trials=np.asarray([1]),
        reserved_rows=np.asarray([], dtype=int),
    )
    path = tmp_path / "partial.csv"
    probabilities = np.eye(6, dtype=float)[[0, 1]]
    row = _metric_and_append(
        path,
        method="hybrid_ensemble",
        target="s05",
        target_index=0,
        split_seed=0,
        model_seed=-1,
        split=split,
        probabilities=probabilities,
        raw_labels_target=np.asarray([0, 0, 1]),
        training_labels_target=np.asarray([0, 0, 1]),
        target_trials=np.asarray([0, 1, 1]),
        n_source_windows=10,
        n_source_subjects=2,
        adaptation_stages="context",
        decoding_mode="offline_context_ensemble",
    )
    assert row["offline_uses_future_windows"] is True


def test_paired_statistics_include_source_only_and_single_model_references() -> None:
    rows = pd.DataFrame(
        [
            {
                "method": method,
                "k_trials_per_sequence": 10,
                "target": target,
                "accuracy_raw_labels": accuracy,
            }
            for target, values in {
                "s05": (0.20, 0.40, 0.50),
                "s06": (0.30, 0.45, 0.55),
            }.items()
            for method, accuracy in zip(
                (
                    "hierarchical_source_only",
                    "hierarchical_tcn_single",
                    "hybrid_ensemble_structured",
                ),
                values,
                strict=True,
            )
        ]
    )
    paired = _paired_push_statistics(rows)
    hybrid = paired[paired["method"] == "hybrid_ensemble_structured"]
    assert set(hybrid["reference_method"]) == {
        "hierarchical_source_only",
        "hierarchical_tcn_single",
    }


def test_curve_summary_uses_one_common_target_cohort_across_k() -> None:
    rows = pd.DataFrame(
        [
            {
                "method": method,
                "k_trials_per_sequence": k,
                "target": target,
                "seed": seed,
                "accuracy_raw_labels": accuracy,
            }
            for method in ("source", "adapted")
            for k in (1, 20)
            for target in (("s05", "s06") if k == 1 else ("s05",))
            for seed in (0, 1)
            for accuracy in [
                0.3
                + 0.1 * (method == "adapted")
                + 0.05 * (k == 20)
                + 0.2 * (target == "s06")
            ]
        ]
    )
    subject, summary, julia, targets = _common_curve_cohort_summaries(rows)
    assert targets == ("s05",)
    assert set(subject["target"]) == {"s05"}
    assert set(summary["n_subjects"]) == {1}
    assert set(julia["n_subject_seed_folds"]) == {2}
    source = summary[summary["method"] == "source"].set_index(
        "k_trials_per_sequence"
    )
    assert source.loc[1, "mean_accuracy_raw_labels"] == pytest.approx(0.3)
    assert source.loc[20, "mean_accuracy_raw_labels"] == pytest.approx(0.35)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
@pytest.mark.parametrize("adapter_kind", ["low_rank", "channel_affine_residual"])
def test_hierarchical_model_rejects_unregistered_target_labels(adapter_kind) -> None:
    from neureptrace.decoding.progressive_temporal_window_finetune import (
        TorchProgressiveTemporalWindowClassifier,
    )

    rng = np.random.default_rng(4)
    windows = rng.normal(size=(36, 4, 3)).astype(np.float32)
    domains = np.repeat([0, 1, 2], 12)
    trials = np.repeat(np.arange(12), 3)
    finger = np.tile(np.arange(6), 6)
    sequence = np.tile(np.arange(3), 12)
    order = finger.copy()
    overlap = (finger > 0).astype(np.float32)
    ratios = np.eye(6, dtype=np.float32)[finger]
    source = np.arange(24)
    masked_finger = finger.copy()
    masked_finger[24:] = 0
    masked_sequence = sequence.copy()
    masked_sequence[24:] = 0
    masked_order = order.copy()
    masked_order[24:] = 0
    masked_overlap = overlap.copy()
    masked_overlap[24:] = 0
    masked_ratios = ratios.copy()
    masked_ratios[24:] = np.asarray([1, 0, 0, 0, 0, 0])
    means = np.stack([windows[domains == domain].mean(axis=(0, 1)) for domain in range(3)])
    stds = np.stack([windows[domains == domain].std(axis=(0, 1)) for domain in range(3)])
    model = TorchProgressiveTemporalWindowClassifier(
        hidden_units=8,
        num_blocks=1,
        adapter_rank=2,
        adapter_kind=adapter_kind,
        source_epochs=1,
        adapter_steps=1,
        last_block_steps=0,
        full_finetune_steps=0,
        batch_size=8,
        hierarchical=True,
        balanced_sampling=True,
        subject_specific_normalization=True,
        source_specific_adapters=True,
        source_selection_metric="finger_accuracy",
        random_state=3,
        device="cpu",
    ).fit_source(
        windows,
        source_indices=source,
        source_domains=domains,
        finger_labels=masked_finger,
        sequence_labels=masked_sequence,
        order_labels=masked_order,
        overlap_targets=masked_overlap,
        sensor_mean=windows[source].mean(axis=(0, 1)),
        sensor_std=windows[source].std(axis=(0, 1)),
        press_ratios=masked_ratios,
        trial_ids=trials,
        subject_sensor_domains=np.arange(3),
        subject_sensor_means=means,
        subject_sensor_stds=stds,
    )
    if adapter_kind == "low_rank":
        import torch

        source_effective = torch.stack(
            [
                model.model_.adapter_scale
                * adapter["up"].weight
                @ adapter["down"].weight
                for adapter in model.model_.source_adapters
            ]
        ).mean(dim=0)
        left, singular_values, right = torch.linalg.svd(
            source_effective, full_matrices=False
        )
        rank = model.model_.adapter_down.weight.shape[0]
        expected = (left[:, :rank] * singular_values[:rank]) @ right[:rank]
        target_effective = (
            model.model_.adapter_scale
            * model.model_.adapter_up.weight
            @ model.model_.adapter_down.weight
        )
        torch.testing.assert_close(target_effective, expected, atol=1e-5, rtol=1e-5)
    clone = model.clone_source(random_state=5)
    trainable = clone._set_stage_trainable("full")
    assert not any(name.startswith("source_") for name in trainable)
    calibration = np.arange(24, 30)
    clone.register_target_calibration_labels(
        calibration,
        finger_labels=finger[calibration],
        sequence_labels=sequence[calibration],
        order_labels=order[calibration],
        overlap_targets=overlap[calibration],
        press_ratios=ratios[calibration],
    ).adapt_target_indices(calibration, n_calibration_trials=2, mode="adapter_only")
    probabilities = clone.predict_proba_indices(np.arange(30, 36))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    with pytest.raises(RuntimeError, match="outside source/calibration"):
        clone._labels_for(np.arange(30, 36))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_causal_trial_transformer_cannot_use_future_embeddings() -> None:
    from neureptrace.decoding.katja_trial_context import TorchKatjaTrialContextRefiner

    rng = np.random.default_rng(22)
    n_rows = 36
    embeddings = rng.normal(size=(n_rows, 6)).astype(np.float32)
    domains = np.repeat([0, 1, 2], 12)
    trials = np.tile(np.repeat([0, 1], 6), 3)
    finger = np.tile(np.arange(6), 6)
    order = finger.copy()
    overlap = (finger > 0).astype(np.float32)
    ratios = np.eye(6, dtype=np.float32)[finger]
    templates = np.tile(np.repeat([0, 1], 6), 3)
    source = np.arange(24)
    masked_finger = finger.copy()
    masked_finger[24:] = 0
    masked_order = order.copy()
    masked_order[24:] = 0
    masked_overlap = overlap.copy()
    masked_overlap[24:] = 0
    masked_ratios = ratios.copy()
    masked_ratios[24:] = np.asarray([1, 0, 0, 0, 0, 0])
    masked_templates = templates.copy()
    masked_templates[24:] = 0
    model = TorchKatjaTrialContextRefiner(
        hidden_units=8,
        num_layers=2,
        num_heads=2,
        source_epochs=1,
        adaptation_steps=1,
        batch_trials=2,
        causal=True,
        random_state=2,
        device="cpu",
    ).fit_source(
        embeddings,
        source_indices=source,
        domain_ids=domains,
        trial_ids=trials,
        finger_labels=masked_finger,
        press_ratios=masked_ratios,
        order_labels=masked_order,
        overlap_targets=masked_overlap,
        template_labels=masked_templates,
    )
    assert model.metadata()["source_refit_all"] is True
    assert model.metadata()["best_source_epoch"] == 1
    calibration = np.arange(24, 30)
    adapted = model.clone_source(random_state=4)
    adapted.embeddings_ = embeddings.copy()
    adapted.register_target_calibration_labels(
        calibration,
        finger_labels=finger[calibration],
        press_ratios=ratios[calibration],
        order_labels=order[calibration],
        overlap_targets=overlap[calibration],
        template_labels=templates[calibration],
    ).adapt_target_indices(calibration)
    evaluation = np.arange(30, 36)
    first = adapted.predict_outputs_indices(evaluation)["probabilities"]
    adapted.embeddings_[evaluation[3:]] += 100.0
    second = adapted.predict_outputs_indices(evaluation)["probabilities"]
    np.testing.assert_allclose(first[:3], second[:3], atol=1e-6)
