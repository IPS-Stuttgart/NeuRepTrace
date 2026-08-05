"""Torch implementation for progressive sequence target adaptation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import copy
import numpy as np

from neureptrace.decoding._progressive_sequence_core import (
    PROGRESSIVE_SEQUENCE_CATEGORY,
    PROGRESSIVE_SEQUENCE_PROTOCOL,
    _as_feature_tensor,
    _as_label_matrix,
    _as_object_vector,
    _bounded_float,
    _labels_equal,
    _nonnegative_float,
    _nonnegative_int,
    _positive_float,
    _positive_int,
    _stable_seed,
    _torch,
    _unique_in_order,
    permutation_constrained_decode,
)
from neureptrace.decoding._progressive_sequence_network import (
    _SequenceAdapterModule,
    _source_trial_validation_split,
    _target_trial_validation_split,
)
from neureptrace.decoding._progressive_sequence_training import _ProgressiveSequenceTrainingMixin


class TorchProgressiveSequenceClassifier(_ProgressiveSequenceTrainingMixin):
    """Sequence-aware source pretraining and progressive target fine-tuning."""

    def __init__(
        self,
        hidden_units: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_multiplier: float = 2.0,
        adapter_rank: int = 8,
        adapter_alpha: float = 8.0,
        source_max_epochs: int = 120,
        source_learning_rate: float = 1e-3,
        adapter_steps: int = 80,
        last_block_steps: int = 60,
        full_finetune_steps: int = 60,
        adaptation_learning_rate: float = 2e-3,
        full_finetune_learning_rate: float = 2e-4,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        dropout: float = 0.15,
        feature_noise_std: float = 0.02,
        feature_dropout: float = 0.05,
        sinkhorn_loss_weight: float = 0.5,
        assignment_loss_weight: float = 0.1,
        source_vrex_weight: float = 0.05,
        meta_epochs: int = 2,
        meta_support_trials: int = 4,
        meta_query_trials: int = 4,
        meta_inner_steps: int = 5,
        meta_learning_rate: float = 2e-3,
        meta_step_size: float = 0.15,
        l2sp_weight: float = 1e-4,
        source_replay_weight: float = 0.1,
        validation_fraction: float = 0.2,
        patience: int = 12,
        min_trials_for_last_block: int = 8,
        min_trials_for_full_finetune: int = 12,
        sinkhorn_temperature: float = 1.0,
        sinkhorn_iterations: int = 20,
        enforce_permutation_labels: bool = True,
        random_state: int | None = 13,
        device: str = "auto",
    ):
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_multiplier = feedforward_multiplier
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.source_max_epochs = source_max_epochs
        self.source_learning_rate = source_learning_rate
        self.adapter_steps = adapter_steps
        self.last_block_steps = last_block_steps
        self.full_finetune_steps = full_finetune_steps
        self.adaptation_learning_rate = adaptation_learning_rate
        self.full_finetune_learning_rate = full_finetune_learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.feature_noise_std = feature_noise_std
        self.feature_dropout = feature_dropout
        self.sinkhorn_loss_weight = sinkhorn_loss_weight
        self.assignment_loss_weight = assignment_loss_weight
        self.source_vrex_weight = source_vrex_weight
        self.meta_epochs = meta_epochs
        self.meta_support_trials = meta_support_trials
        self.meta_query_trials = meta_query_trials
        self.meta_inner_steps = meta_inner_steps
        self.meta_learning_rate = meta_learning_rate
        self.meta_step_size = meta_step_size
        self.l2sp_weight = l2sp_weight
        self.source_replay_weight = source_replay_weight
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_trials_for_last_block = min_trials_for_last_block
        self.min_trials_for_full_finetune = min_trials_for_full_finetune
        self.sinkhorn_temperature = sinkhorn_temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.enforce_permutation_labels = enforce_permutation_labels
        self.random_state = random_state
        self.device = device

    def fit_source(
        self,
        source_features: Sequence | np.ndarray,
        source_labels: Sequence | np.ndarray,
        *,
        source_subjects: Sequence[Any] | np.ndarray | None = None,
    ) -> "TorchProgressiveSequenceClassifier":
        torch = _torch()
        x = _as_feature_tensor(source_features, name="source_features")
        labels = _as_label_matrix(source_labels, name="source_labels")
        if labels.shape != x.shape[:2]:
            raise ValueError("source_labels must match the source trial and event dimensions.")
        self.classes_, y = np.unique(labels, return_inverse=True)
        y = y.reshape(labels.shape).astype(np.int64, copy=False)
        if self.classes_.shape[0] < 2:
            raise ValueError("Source training needs at least two classes.")
        self.n_events_ = int(x.shape[1])
        self.n_features_in_ = int(x.shape[2])
        self.n_classes_ = int(self.classes_.shape[0])
        self._validate_permutation_labels(y, name="source_labels")

        if source_subjects is None:
            subjects = np.asarray(["source"] * x.shape[0], dtype=object)
        else:
            subjects = _as_object_vector(source_subjects, name="source_subjects")
            if subjects.shape[0] != x.shape[0]:
                raise ValueError("source_subjects must contain one value per source trial.")
        subject_names, domains = np.unique(subjects.astype(str), return_inverse=True)
        domains = domains.astype(np.int64, copy=False)
        self.source_subjects_ = subject_names

        seed = None if self.random_state is None else _nonnegative_int(self.random_state, "random_state")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.device_ = self._resolve_device()
        hidden_units = _positive_int(self.hidden_units, "hidden_units")
        num_heads = _positive_int(self.num_heads, "num_heads")
        if hidden_units % num_heads:
            raise ValueError("hidden_units must be divisible by num_heads.")
        self.model_ = _SequenceAdapterModule(
            input_dim=self.n_features_in_,
            n_events=self.n_events_,
            n_classes=self.n_classes_,
            hidden_units=hidden_units,
            num_layers=_positive_int(self.num_layers, "num_layers"),
            num_heads=num_heads,
            feedforward_multiplier=_positive_float(self.feedforward_multiplier, "feedforward_multiplier"),
            adapter_rank=_positive_int(self.adapter_rank, "adapter_rank"),
            adapter_alpha=_positive_float(self.adapter_alpha, "adapter_alpha"),
            dropout=_bounded_float(self.dropout, "dropout", lower=0.0, upper=1.0),
        ).to(self.device_)
        self._set_source_trainable()
        train_idx, validation_idx, validation_mode = _source_trial_validation_split(
            y,
            domains,
            validation_fraction=_bounded_float(self.validation_fraction, "validation_fraction", lower=0.0, upper=1.0),
            random_state=seed,
        )
        self._train_source(x, y, domains, train_idx=train_idx, validation_idx=validation_idx)
        self.source_validation_mode_ = validation_mode
        self.meta_episodes_run_, self.meta_episodes_accepted_ = self._meta_initialize_target_adapter(x, y, domains)
        self.source_features_ = x.copy()
        self.source_encoded_labels_ = y.copy()
        self.source_domains_encoded_ = domains.copy()
        self.source_state_ = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
        self.model_.eval()
        return self

    def adapt_target(
        self,
        target_calibration_features: Sequence | np.ndarray,
        target_calibration_labels: Sequence | np.ndarray,
        *,
        target_strata: Sequence[Any] | np.ndarray | None = None,
    ) -> "TorchProgressiveSequenceClassifier":
        if not hasattr(self, "source_state_"):
            raise RuntimeError("fit_source must be called before adapt_target.")
        x = _as_feature_tensor(target_calibration_features, name="target_calibration_features")
        labels = _as_label_matrix(target_calibration_labels, name="target_calibration_labels")
        if x.shape[1:] != (self.n_events_, self.n_features_in_) or labels.shape != x.shape[:2]:
            raise ValueError("Target calibration tensors must match the fitted source event and feature dimensions.")
        y = self._encode_target_labels(labels)
        self._validate_permutation_labels(y, name="target_calibration_labels")
        strata = None if target_strata is None else _as_object_vector(target_strata, name="target_strata")
        if strata is not None and strata.shape[0] != x.shape[0]:
            raise ValueError("target_strata must contain one value per target calibration trial.")

        self.model_.load_state_dict(self.source_state_)
        train_idx, validation_idx, validation_mode = _target_trial_validation_split(
            x.shape[0],
            strata=strata,
            validation_fraction=_bounded_float(self.validation_fraction, "validation_fraction", lower=0.0, upper=1.0),
            random_state=self.random_state,
        )
        self.target_validation_mode_ = validation_mode
        self.adaptation_stage_history_ = []
        stages: list[tuple[str, int, float]] = [
            ("adapter", _nonnegative_int(self.adapter_steps, "adapter_steps"), _positive_float(self.adaptation_learning_rate, "adaptation_learning_rate")),
        ]
        if x.shape[0] >= _positive_int(self.min_trials_for_last_block, "min_trials_for_last_block"):
            stages.append(
                ("last_block", _nonnegative_int(self.last_block_steps, "last_block_steps"), _positive_float(self.adaptation_learning_rate, "adaptation_learning_rate"))
            )
        if x.shape[0] >= _positive_int(self.min_trials_for_full_finetune, "min_trials_for_full_finetune"):
            stages.append(
                ("full", _nonnegative_int(self.full_finetune_steps, "full_finetune_steps"), _positive_float(self.full_finetune_learning_rate, "full_finetune_learning_rate"))
            )
        for stage_name, steps, learning_rate in stages:
            if steps == 0:
                continue
            self._adapt_stage(
                x,
                y,
                train_idx=train_idx,
                validation_idx=validation_idx,
                stage=stage_name,
                steps=steps,
                learning_rate=learning_rate,
            )
        self.target_calibration_trials_ = int(x.shape[0])
        self.target_train_trials_ = int(train_idx.shape[0])
        self.target_validation_trials_ = int(validation_idx.shape[0])
        self.model_.eval()
        return self

    def predict_proba(self, features: Sequence | np.ndarray, *, constrained: bool = False) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("The decoder must be fitted before prediction.")
        torch = _torch()
        x = _as_feature_tensor(features, name="features")
        if x.shape[1:] != (self.n_events_, self.n_features_in_):
            raise ValueError("features must match the fitted event and feature dimensions.")
        self.model_.eval()
        probabilities: list[np.ndarray] = []
        batch_size = _positive_int(self.batch_size, "batch_size")
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                tensor = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device_)
                logits = self.model_(tensor, use_adapter=True)
                probabilities.append(torch.softmax(logits, dim=2).detach().cpu().numpy())
        result = np.concatenate(probabilities, axis=0).astype(float, copy=False)
        if constrained:
            return permutation_constrained_decode(
                result,
                temperature=self.sinkhorn_temperature,
                sinkhorn_iterations=self.sinkhorn_iterations,
            ).probabilities
        return result

    def predict(self, features: Sequence | np.ndarray, *, constrained: bool = True) -> np.ndarray:
        probabilities = self.predict_proba(features, constrained=False)
        if constrained and self.enforce_permutation_labels:
            encoded = permutation_constrained_decode(
                probabilities,
                temperature=self.sinkhorn_temperature,
                sinkhorn_iterations=self.sinkhorn_iterations,
            ).assignments
        else:
            encoded = np.argmax(probabilities, axis=2)
        return self.classes_[encoded]

    def metadata(self) -> dict[str, Any]:
        return {
            "progressive_sequence_protocol": PROGRESSIVE_SEQUENCE_PROTOCOL,
            "progressive_sequence_protocol_category": PROGRESSIVE_SEQUENCE_CATEGORY,
            "progressive_sequence_uses_target_features": True,
            "progressive_sequence_uses_target_labels": True,
            "progressive_sequence_valid_for_strict_source_only": False,
            "progressive_sequence_valid_for_protocol_3_benchmark": True,
            "progressive_sequence_trial_structure": True,
            "progressive_sequence_permutation_constraint": bool(self.enforce_permutation_labels),
            "progressive_sequence_hidden_units": int(self.hidden_units),
            "progressive_sequence_num_layers": int(self.num_layers),
            "progressive_sequence_num_heads": int(self.num_heads),
            "progressive_sequence_adapter_rank": int(self.adapter_rank),
            "progressive_sequence_source_epochs_run": int(getattr(self, "source_epochs_run_", 0)),
            "progressive_sequence_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "progressive_sequence_source_validation_mode": getattr(self, "source_validation_mode_", "unknown"),
            "progressive_sequence_meta_epochs": int(self.meta_epochs),
            "progressive_sequence_meta_episodes_run": int(getattr(self, "meta_episodes_run_", 0)),
            "progressive_sequence_meta_episodes_accepted": int(getattr(self, "meta_episodes_accepted_", 0)),
            "progressive_sequence_target_validation_mode": getattr(self, "target_validation_mode_", "unknown"),
            "progressive_sequence_target_calibration_trials": int(getattr(self, "target_calibration_trials_", 0)),
            "progressive_sequence_target_train_trials": int(getattr(self, "target_train_trials_", 0)),
            "progressive_sequence_target_validation_trials": int(getattr(self, "target_validation_trials_", 0)),
            "progressive_sequence_adaptation_stages": tuple(item["stage"] for item in getattr(self, "adaptation_stage_history_", [])),
            "progressive_sequence_adaptation_history": copy.deepcopy(getattr(self, "adaptation_stage_history_", [])),
            "progressive_sequence_source_replay_weight": float(self.source_replay_weight),
            "progressive_sequence_l2sp_weight": float(self.l2sp_weight),
            "progressive_sequence_device": str(getattr(self, "device_", self.device)),
        }
