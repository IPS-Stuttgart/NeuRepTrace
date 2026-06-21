"""Semi-supervised LoRA few-shot target calibration for Protocol-3 decoding.

The module is intentionally feature-matrix based.  It trains a small source
neural classifier, optionally meta-initializes its adapter/head parameters with
source-subject episodes, then freezes the source backbone and adapts only LoRA
adapter weights plus the classifier head on a labeled target calibration subset.
Optional unlabeled target features can contribute entropy and consistency losses.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import copy
import hashlib

import numpy as np

from neureptrace.decoding.few_shot import (
    FewShotTargetCalibrationSplit as LoRAFewShotTargetCalibrationSplit,
    _as_1d_object_array,
    _as_feature_matrix,
    select_few_shot_target_calibration_split,
)

LORA_FEW_SHOT_CALIBRATION_PROTOCOL = "semi_supervised_lora_few_shot_calibration"
LORA_FEW_SHOT_CALIBRATION_CATEGORY = "3_supervised_calibrated_target_alignment"


@dataclass(frozen=True, slots=True)
class LoRAFewShotCalibrationResult:
    """Fitted LoRA few-shot decoder and fold-local outputs."""

    model: "TorchLoRAFewShotClassifier"
    probabilities: np.ndarray
    evaluation_indices: np.ndarray
    calibration_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def select_lora_few_shot_target_calibration_split(*args, **kwargs) -> LoRAFewShotTargetCalibrationSplit:
    """Alias for the balanced Protocol-3 target calibration splitter."""

    return select_few_shot_target_calibration_split(*args, **kwargs)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("The LoRA few-shot decoder requires torch, e.g. `pip install neureptrace[torch]`.") from exc
    return torch


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(number)


def _positive_int(value: Any, name: str) -> int:
    integer = _integer(value, name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: Any, name: str) -> int:
    integer = _integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return number


def _bounded_float(value: Any, name: str, *, lower: float, upper: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < lower or number > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return number


def _stable_seed(*parts: Any) -> int:
    payload = repr(tuple(str(part) for part in parts)).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def _encode_labels(labels: np.ndarray, classes: np.ndarray, *, name: str) -> np.ndarray:
    class_to_index = {class_label: index for index, class_label in enumerate(classes.tolist())}
    encoded = []
    for label in labels.tolist():
        if label not in class_to_index:
            raise ValueError(f"{name} contains class {label!r}, which is absent from source classes.")
        encoded.append(class_to_index[label])
    return np.asarray(encoded, dtype=np.int64)


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if probabilities.ndim != 2 or not np.all(np.isfinite(probabilities)) or np.any(row_sums <= 0.0):
        raise ValueError("Predicted probabilities must be finite two-dimensional rows with positive mass.")
    return probabilities / row_sums


class TorchLoRAFewShotClassifier:
    """Small LoRA neural decoder for Protocol-3 few-shot target calibration.

    Source fitting trains the full model.  Target adaptation freezes the base hidden
    layer and updates only low-rank LoRA matrices and, by default, the classifier
    head.  If ``source_subjects`` and ``meta_epochs`` are supplied, source subjects
    are treated as pseudo-targets in Reptile-style episodes before target fitting.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        source_max_epochs: int = 80,
        source_learning_rate: float = 1e-3,
        adaptation_steps: int = 80,
        adaptation_learning_rate: float = 5e-3,
        meta_epochs: int = 0,
        meta_support_per_class: int = 1,
        meta_query_per_class: int = 1,
        meta_inner_steps: int = 5,
        meta_step_size: float = 0.25,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        entropy_loss_weight: float = 0.0,
        consistency_loss_weight: float = 0.0,
        consistency_noise_std: float = 0.01,
        source_replay_weight: float = 0.0,
        adapt_classifier_head: bool = True,
        validation_fraction: float = 0.1,
        patience: int = 8,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
        device: str = "auto",
    ):
        self.hidden_units = hidden_units
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.source_max_epochs = source_max_epochs
        self.source_learning_rate = source_learning_rate
        self.adaptation_steps = adaptation_steps
        self.adaptation_learning_rate = adaptation_learning_rate
        self.meta_epochs = meta_epochs
        self.meta_support_per_class = meta_support_per_class
        self.meta_query_per_class = meta_query_per_class
        self.meta_inner_steps = meta_inner_steps
        self.meta_step_size = meta_step_size
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.entropy_loss_weight = entropy_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_noise_std = consistency_noise_std
        self.source_replay_weight = source_replay_weight
        self.adapt_classifier_head = adapt_classifier_head
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.class_weight = class_weight
        self.device = device

    def fit_source(self, source_features: Sequence[Sequence[float]] | np.ndarray, source_labels: Sequence[Any] | np.ndarray, *, source_subjects: Sequence[Any] | np.ndarray | None = None):
        torch = _torch()
        x = _as_feature_matrix(source_features, name="source_features").astype(np.float32, copy=False)
        y_raw = _as_1d_object_array(source_labels, name="source_labels")
        if x.shape[0] != y_raw.shape[0]:
            raise ValueError("source_features and source_labels must contain the same rows.")
        self.classes_, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64, copy=False)
        if self.classes_.shape[0] < 2:
            raise ValueError("LoRA few-shot source training needs at least two classes.")

        seed = _nonnegative_int(self.random_state, "lora_few_shot_random_state") if self.random_state is not None else None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.device_ = self._resolve_device()
        self.n_features_in_ = int(x.shape[1])
        self.model_ = _LoRAMLPModule(
            input_dim=x.shape[1],
            hidden_units=_positive_int(self.hidden_units, "lora_few_shot_hidden_units"),
            n_classes=int(self.classes_.shape[0]),
            lora_rank=_positive_int(self.lora_rank, "lora_few_shot_lora_rank"),
            lora_alpha=_positive_float(self.lora_alpha, "lora_few_shot_lora_alpha"),
            dropout=_bounded_float(self.dropout, "lora_few_shot_dropout", lower=0.0, upper=1.0),
        ).to(self.device_)
        train_idx, validation_idx = _stratified_train_validation_indices(
            y,
            validation_fraction=_bounded_float(self.validation_fraction, "lora_few_shot_validation_fraction", lower=0.0, upper=1.0),
            random_state=seed,
        )
        self._train_supervised(
            x,
            y,
            train_idx=train_idx,
            validation_idx=validation_idx,
            epochs=_positive_int(self.source_max_epochs, "lora_few_shot_source_max_epochs"),
            learning_rate=_positive_float(self.source_learning_rate, "lora_few_shot_source_learning_rate"),
            train_all=True,
        )
        self.source_features_ = x.copy()
        self.source_encoded_labels_ = y.copy()
        self.meta_episodes_run_ = 0
        if source_subjects is not None and _nonnegative_int(self.meta_epochs, "lora_few_shot_meta_epochs") > 0:
            subjects = _as_1d_object_array(source_subjects, name="source_subjects")
            if subjects.shape[0] != x.shape[0]:
                raise ValueError("source_subjects must contain one value per source row.")
            self.meta_episodes_run_ = self._run_reptile_meta_training(x, y, subjects, random_state=seed)
        return self

    def adapt_target(
        self,
        target_calibration_features: Sequence[Sequence[float]] | np.ndarray,
        target_calibration_labels: Sequence[Any] | np.ndarray,
        *,
        target_unlabeled_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    ):
        if not hasattr(self, "model_"):
            raise RuntimeError("fit_source must be called before adapt_target.")
        torch = _torch()
        x_cal = _as_feature_matrix(target_calibration_features, name="target_calibration_features").astype(np.float32, copy=False)
        y_cal_raw = _as_1d_object_array(target_calibration_labels, name="target_calibration_labels")
        if x_cal.shape[0] != y_cal_raw.shape[0] or x_cal.shape[1] != self.n_features_in_:
            raise ValueError("target calibration features/labels must match rows and source feature width.")
        y_cal = _encode_labels(y_cal_raw, self.classes_, name="target_calibration_labels")
        if np.unique(y_cal).shape[0] < min(2, self.classes_.shape[0]):
            raise ValueError("target calibration must contain at least two classes for LoRA few-shot adaptation.")
        x_unlabeled = None
        if target_unlabeled_features is not None:
            x_unlabeled = _as_feature_matrix(target_unlabeled_features, name="target_unlabeled_features").astype(np.float32, copy=False)
            if x_unlabeled.shape[1] != self.n_features_in_:
                raise ValueError("target_unlabeled_features must have the source feature width.")

        self._set_adaptation_trainable(self.model_)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model_.parameters() if parameter.requires_grad],
            lr=_positive_float(self.adaptation_learning_rate, "lora_few_shot_adaptation_learning_rate"),
            weight_decay=_nonnegative_float(self.weight_decay, "lora_few_shot_weight_decay"),
        )
        x_cal_tensor = torch.as_tensor(x_cal, dtype=torch.float32, device=self.device_)
        y_cal_tensor = torch.as_tensor(y_cal, dtype=torch.long, device=self.device_)
        x_unlabeled_tensor = None if x_unlabeled is None else torch.as_tensor(x_unlabeled, dtype=torch.float32, device=self.device_)
        source_tensor = torch.as_tensor(self.source_features_, dtype=torch.float32, device=self.device_)
        source_label_tensor = torch.as_tensor(self.source_encoded_labels_, dtype=torch.long, device=self.device_)
        entropy_weight = _nonnegative_float(self.entropy_loss_weight, "lora_few_shot_entropy_loss_weight")
        consistency_weight = _nonnegative_float(self.consistency_loss_weight, "lora_few_shot_consistency_loss_weight")
        replay_weight = _nonnegative_float(self.source_replay_weight, "lora_few_shot_source_replay_weight")
        noise_std = _nonnegative_float(self.consistency_noise_std, "lora_few_shot_consistency_noise_std")
        batch_size = _positive_int(self.batch_size, "lora_few_shot_batch_size")
        rng = np.random.default_rng(self.random_state)
        loss_fn = self._loss_fn(y_cal)
        best_loss = np.inf
        best_state = None
        for _step in range(_positive_int(self.adaptation_steps, "lora_few_shot_adaptation_steps")):
            self.model_.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model_(x_cal_tensor), y_cal_tensor)
            if replay_weight > 0.0:
                replay_size = min(batch_size, self.source_encoded_labels_.shape[0])
                replay_idx = rng.choice(self.source_encoded_labels_.shape[0], size=replay_size, replace=self.source_encoded_labels_.shape[0] < replay_size)
                loss = loss + replay_weight * self._loss_fn(self.source_encoded_labels_[replay_idx])(self.model_(source_tensor[replay_idx]), source_label_tensor[replay_idx])
            if x_unlabeled_tensor is not None and entropy_weight > 0.0:
                probabilities = torch.softmax(self.model_(x_unlabeled_tensor), dim=1)
                loss = loss + entropy_weight * (-(probabilities * torch.log(torch.clamp(probabilities, min=1e-8))).sum(dim=1).mean())
            if x_unlabeled_tensor is not None and consistency_weight > 0.0:
                clean_logits = self.model_(x_unlabeled_tensor)
                noisy_logits = self.model_(x_unlabeled_tensor + torch.randn_like(x_unlabeled_tensor) * noise_std)
                loss = loss + consistency_weight * torch.nn.functional.mse_loss(torch.softmax(noisy_logits, dim=1), torch.softmax(clean_logits.detach(), dim=1))
            loss.backward()
            optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                calibration_loss = float(loss_fn(self.model_(x_cal_tensor), y_cal_tensor).detach().cpu())
            if calibration_loss + 1e-6 < best_loss:
                best_loss = calibration_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.target_adaptation_loss_ = float(best_loss)
        self.target_calibration_rows_ = int(x_cal.shape[0])
        self.target_unlabeled_rows_ = 0 if x_unlabeled is None else int(x_unlabeled.shape[0])
        self.adaptation_trainable_parameter_names_ = tuple(name for name, parameter in self.model_.named_parameters() if parameter.requires_grad)
        self.model_.eval()
        return self

    def predict_proba(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        torch = _torch()
        x = _as_feature_matrix(features, name="features").astype(np.float32, copy=False)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("features must have the fitted feature width.")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(x, dtype=torch.float32, device=self.device_))
        return _normalize_probability_rows(torch.softmax(logits, dim=1).detach().cpu().numpy())

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def metadata(self) -> dict[str, Any]:
        return {
            "lora_few_shot_protocol": LORA_FEW_SHOT_CALIBRATION_PROTOCOL,
            "lora_few_shot_protocol_category": LORA_FEW_SHOT_CALIBRATION_CATEGORY,
            "lora_few_shot_uses_target_features": True,
            "lora_few_shot_uses_target_labels": True,
            "lora_few_shot_uses_unlabeled_target_features": bool(getattr(self, "target_unlabeled_rows_", 0) > 0),
            "lora_few_shot_valid_for_strict_source_only": False,
            "lora_few_shot_valid_for_unlabeled_target_adaptation": False,
            "lora_few_shot_valid_for_protocol_3_benchmark": True,
            "lora_few_shot_hidden_units": int(self.hidden_units),
            "lora_few_shot_lora_rank": int(self.lora_rank),
            "lora_few_shot_lora_alpha": float(self.lora_alpha),
            "lora_few_shot_source_epochs_run": int(getattr(self, "source_epochs_run_", 0)),
            "lora_few_shot_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "lora_few_shot_adaptation_steps": int(self.adaptation_steps),
            "lora_few_shot_adaptation_loss": float(getattr(self, "target_adaptation_loss_", np.nan)),
            "lora_few_shot_meta_learning": bool(int(self.meta_epochs) > 0),
            "lora_few_shot_meta_epochs": int(self.meta_epochs),
            "lora_few_shot_meta_episodes_run": int(getattr(self, "meta_episodes_run_", 0)),
            "lora_few_shot_entropy_loss_weight": float(self.entropy_loss_weight),
            "lora_few_shot_consistency_loss_weight": float(self.consistency_loss_weight),
            "lora_few_shot_source_replay_weight": float(self.source_replay_weight),
            "lora_few_shot_target_calibration_rows": int(getattr(self, "target_calibration_rows_", 0)),
            "lora_few_shot_target_unlabeled_rows": int(getattr(self, "target_unlabeled_rows_", 0)),
            "lora_few_shot_trainable_parameter_names": tuple(getattr(self, "adaptation_trainable_parameter_names_", ())),
            "lora_few_shot_device": str(getattr(self, "device_", self.device)),
        }

    def _resolve_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _loss_fn(self, encoded_labels: np.ndarray):
        torch = _torch()
        if self.class_weight == "balanced":
            counts = np.bincount(np.asarray(encoded_labels, dtype=np.int64), minlength=int(self.classes_.shape[0])).astype(np.float32)
            weights = encoded_labels.shape[0] / np.maximum(counts, 1.0) / float(self.classes_.shape[0])
            return torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device_))
        return torch.nn.CrossEntropyLoss()

    def _set_adaptation_trainable(self, model) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("lora_") or (bool(self.adapt_classifier_head) and name.startswith("class_head")))

    def _train_supervised(self, x: np.ndarray, y: np.ndarray, *, train_idx: np.ndarray, validation_idx: np.ndarray, epochs: int, learning_rate: float, train_all: bool) -> None:
        torch = _torch()
        if train_all:
            for parameter in self.model_.parameters():
                parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter for parameter in self.model_.parameters() if parameter.requires_grad], lr=learning_rate, weight_decay=_nonnegative_float(self.weight_decay, "lora_few_shot_weight_decay"))
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device_)
        loss_fn = self._loss_fn(y[train_idx])
        rng = np.random.default_rng(self.random_state)
        best_loss = np.inf
        best_state = None
        patience_left = _positive_int(self.patience, "lora_few_shot_patience")
        batch_size = _positive_int(self.batch_size, "lora_few_shot_batch_size")
        self.source_epochs_run_ = 0
        for epoch in range(epochs):
            self.source_epochs_run_ = epoch + 1
            self.model_.train()
            for start in range(0, train_idx.shape[0], batch_size):
                batch_idx = rng.permutation(train_idx)[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(x_tensor[batch_idx]), y_tensor[batch_idx])
                loss.backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(self.model_(x_tensor[validation_idx]), y_tensor[validation_idx]).detach().cpu())
            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                patience_left = _positive_int(self.patience, "lora_few_shot_patience")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_source_validation_loss_ = float(best_loss)

    def _run_reptile_meta_training(self, x: np.ndarray, y: np.ndarray, subjects: np.ndarray, *, random_state: int | None) -> int:
        torch = _torch()
        episodes_run = 0
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device_)
        rng = np.random.default_rng(random_state)
        for epoch in range(_positive_int(self.meta_epochs, "lora_few_shot_meta_epochs")):
            for subject in rng.permutation(np.asarray(list(dict.fromkeys(subjects.tolist())), dtype=object)):
                support_idx, _query_idx = _balanced_subject_episode_indices(
                    y,
                    subjects,
                    subject,
                    support_per_class=_positive_int(self.meta_support_per_class, "lora_few_shot_meta_support_per_class"),
                    query_per_class=_positive_int(self.meta_query_per_class, "lora_few_shot_meta_query_per_class"),
                    seed=_stable_seed(random_state, epoch, episodes_run, subject),
                )
                if support_idx.size == 0:
                    continue
                clone = copy.deepcopy(self.model_).to(self.device_)
                self._set_adaptation_trainable(clone)
                optimizer = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad], lr=_positive_float(self.adaptation_learning_rate, "lora_few_shot_adaptation_learning_rate"), weight_decay=_nonnegative_float(self.weight_decay, "lora_few_shot_weight_decay"))
                loss_fn = self._loss_fn(y[support_idx])
                for _inner in range(_positive_int(self.meta_inner_steps, "lora_few_shot_meta_inner_steps")):
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(clone(x_tensor[support_idx]), y_tensor[support_idx])
                    loss.backward()
                    optimizer.step()
                step_size = _bounded_float(self.meta_step_size, "lora_few_shot_meta_step_size", lower=0.0, upper=1.0)
                with torch.no_grad():
                    clone_parameters = dict(clone.named_parameters())
                    for name, parameter in self.model_.named_parameters():
                        if parameter.requires_grad and name in clone_parameters:
                            parameter.add_(step_size * (clone_parameters[name].detach() - parameter))
                episodes_run += 1
        return int(episodes_run)


def _LoRAMLPModule(*, input_dim: int, hidden_units: int, n_classes: int, lora_rank: int, lora_alpha: float, dropout: float):
    torch = _torch()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(input_dim, hidden_units)
            self.lora_a = torch.nn.Linear(input_dim, lora_rank, bias=False)
            self.lora_b = torch.nn.Linear(lora_rank, hidden_units, bias=False)
            self.class_head = torch.nn.Linear(hidden_units, n_classes)
            self.dropout = torch.nn.Dropout(float(dropout))
            self.scale = float(lora_alpha) / float(max(lora_rank, 1))
            torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
            torch.nn.init.zeros_(self.lora_b.weight)

        def forward(self, features):
            hidden = self.base(features) + self.scale * self.lora_b(self.lora_a(features))
            return self.class_head(self.dropout(torch.relu(hidden)))

    return Module()


def _stratified_train_validation_indices(y: np.ndarray, *, validation_fraction: float, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(y.shape[0], dtype=int)
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        return indices, indices
    rng = np.random.default_rng(random_state)
    train_parts = []
    validation_parts = []
    for class_label in np.unique(y):
        class_indices = np.flatnonzero(y == class_label)
        if class_indices.size < 2:
            return indices, indices
        shuffled = rng.permutation(class_indices)
        n_validation = min(max(1, int(round(class_indices.size * validation_fraction))), class_indices.size - 1)
        validation_parts.append(shuffled[:n_validation])
        train_parts.append(shuffled[n_validation:])
    return np.sort(np.concatenate(train_parts).astype(int, copy=False)), np.sort(np.concatenate(validation_parts).astype(int, copy=False))


def _balanced_subject_episode_indices(y: np.ndarray, subjects: np.ndarray, subject: Any, *, support_per_class: int, query_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    support_parts = []
    query_parts = []
    for class_label in np.unique(y):
        positions = np.flatnonzero((subjects == subject) & (y == class_label))
        required = int(support_per_class) + int(query_per_class)
        if positions.size < required:
            return np.array([], dtype=int), np.array([], dtype=int)
        shuffled = rng.permutation(positions)
        support_parts.append(shuffled[:support_per_class])
        query_parts.append(shuffled[support_per_class:required])
    return np.sort(np.concatenate(support_parts).astype(int, copy=False)), np.sort(np.concatenate(query_parts).astype(int, copy=False))


def fit_lora_few_shot_target_calibrated_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    source_subjects: Sequence[Any] | np.ndarray | None = None,
    target_unlabeled_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    use_evaluation_features_as_unlabeled: bool = False,
    split: LoRAFewShotTargetCalibrationSplit | None = None,
    per_class: int | str = 1,
    seed: int | str = 13,
    context: Sequence[Hashable] = (),
    min_evaluation_per_class: int | str = 1,
    **model_kwargs: Any,
) -> LoRAFewShotCalibrationResult:
    """Fit a source-meta-trained LoRA decoder and predict held-out target rows.

    Target labels are used only for calibration-row selection and calibration loss.
    Evaluation labels must not be used.  Setting
    ``use_evaluation_features_as_unlabeled=True`` enables an explicitly flagged
    transductive variant that adapts with evaluation features but not labels.
    """

    source_matrix = _as_feature_matrix(source_features, name="source_features")
    target_matrix = _as_feature_matrix(target_features, name="target_features")
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError("source_features and target_features must have the same feature width.")
    source_label_vector = _as_1d_object_array(source_labels, name="source_labels")
    target_label_vector = _as_1d_object_array(target_labels, name="target_labels")
    if source_matrix.shape[0] != source_label_vector.shape[0] or target_matrix.shape[0] != target_label_vector.shape[0]:
        raise ValueError("features and labels must have matching row counts.")
    seed_value = _nonnegative_int(seed, "lora_few_shot_target_calibration_seed")
    per_class_count = _positive_int(per_class, "lora_few_shot_target_calibration_per_class")
    if split is None:
        split = select_lora_few_shot_target_calibration_split(target_label_vector, per_class=per_class_count, seed=seed_value, context=context, min_evaluation_per_class=min_evaluation_per_class)
    calibration_indices = np.asarray(split.calibration_indices, dtype=int).reshape(-1)
    evaluation_indices = np.asarray(split.evaluation_indices, dtype=int).reshape(-1)
    if calibration_indices.size == 0 or evaluation_indices.size == 0:
        raise ValueError("LoRA few-shot calibration requires non-empty calibration and evaluation rows.")
    for name, indices in {"calibration_indices": calibration_indices, "evaluation_indices": evaluation_indices}.items():
        if np.any(indices < 0) or np.any(indices >= target_matrix.shape[0]):
            raise ValueError(f"{name} contains an out-of-range target row index.")
    if np.intersect1d(calibration_indices, evaluation_indices).size:
        raise ValueError("LoRA few-shot calibration and evaluation indices must be disjoint.")

    unlabeled_features = None if target_unlabeled_features is None else _as_feature_matrix(target_unlabeled_features, name="target_unlabeled_features")
    if unlabeled_features is not None and unlabeled_features.shape[1] != source_matrix.shape[1]:
        raise ValueError("target_unlabeled_features must have the same feature width as source_features.")
    if use_evaluation_features_as_unlabeled:
        evaluation_unlabeled = target_matrix[evaluation_indices]
        unlabeled_features = evaluation_unlabeled if unlabeled_features is None else np.vstack([unlabeled_features, evaluation_unlabeled])

    model = TorchLoRAFewShotClassifier(random_state=seed_value, **model_kwargs)
    model.fit_source(source_matrix, source_label_vector, source_subjects=source_subjects)
    model.adapt_target(target_matrix[calibration_indices], target_label_vector[calibration_indices], target_unlabeled_features=unlabeled_features)
    probabilities = model.predict_proba(target_matrix[evaluation_indices])
    metadata = model.metadata()
    metadata.update(
        {
            "lora_few_shot_target_calibration_per_class": int(per_class_count),
            "lora_few_shot_target_calibration_seed": int(seed_value),
            "lora_few_shot_n_source_rows": int(source_matrix.shape[0]),
            "lora_few_shot_n_target_rows": int(target_matrix.shape[0]),
            "lora_few_shot_n_target_calibration_rows": int(calibration_indices.size),
            "lora_few_shot_n_target_evaluation_rows": int(evaluation_indices.size),
            "lora_few_shot_uses_evaluation_features_as_unlabeled": bool(use_evaluation_features_as_unlabeled),
            "lora_few_shot_transductive_evaluation_features": bool(use_evaluation_features_as_unlabeled),
        }
    )
    return LoRAFewShotCalibrationResult(
        model=model,
        probabilities=probabilities,
        evaluation_indices=evaluation_indices.copy(),
        calibration_indices=calibration_indices.copy(),
        metadata=metadata,
    )
