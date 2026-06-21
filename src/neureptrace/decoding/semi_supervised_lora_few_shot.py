"""Semi-supervised meta-learning and LoRA few-shot target calibration.

This module implements a Protocol-3 decoder for held-out target subjects.  The
model is first trained on labeled source subjects, optionally receives a
source-subject episodic Reptile-style meta-learning pass, and is then adapted to a
small labeled calibration subset from the held-out target subject.  Unlabeled
held-out target features may contribute entropy/consistency losses, but their
labels are never accepted or used.
"""

from __future__ import annotations

import copy
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:  # keep module importable when the optional torch extra is absent
    import torch as _TORCH
except ImportError:  # pragma: no cover - depends on optional extra
    _TORCH = None

from neureptrace.decoding.few_shot import FewShotTargetCalibrationSplit, select_few_shot_target_calibration_split

SEMI_SUPERVISED_LORA_FEW_SHOT_PROTOCOL = "semi_supervised_lora_few_shot_calibration"
SEMI_SUPERVISED_LORA_FEW_SHOT_CATEGORY = "3_supervised_calibrated_target_alignment"
SEMI_SUPERVISED_LORA_META_ALGORITHM = "reptile_lora_adapter_initialization"


@dataclass(frozen=True, slots=True)
class SemiSupervisedLoRAFewShotResult:
    """Fitted semi-supervised LoRA few-shot model and target-fold outputs."""

    model: "SemiSupervisedLoRAFewShotClassifier"
    probabilities: np.ndarray
    evaluation_indices: np.ndarray
    calibration_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _torch():
    if _TORCH is None:  # pragma: no cover - depends on optional extra
        raise ImportError("Semi-supervised LoRA few-shot calibration requires torch, e.g. `pip install neureptrace[torch]`.")
    return _TORCH


def _as_feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if not allow_empty and matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _as_optional_feature_matrix(values: Sequence[Sequence[float]] | np.ndarray | None, *, name: str, n_features: int) -> np.ndarray:
    if values is None:
        return np.empty((0, int(n_features)), dtype=float)
    matrix = _as_feature_matrix(values, name=name, allow_empty=True)
    if matrix.shape[1] != n_features:
        raise ValueError(f"{name} must have {n_features} feature columns; got {matrix.shape[1]}.")
    return matrix


def _as_1d_object_array(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array.reshape(-1)


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def _positive_float(value: float | str, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite float.") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite float.")
    return float(numeric)


def _nonnegative_float(value: float | str, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite float.") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite float.")
    return float(numeric)


def _bounded_float(value: float | str, *, name: str, lower: float, upper: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < lower or numeric > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return float(numeric)


def _normalize_adapt_head(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"none", "bias", "full"}:
        raise ValueError("adapt_head must be one of 'none', 'bias', or 'full'.")
    return normalized


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities must be finite.")
    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probability rows must have positive mass.")
    return probabilities / row_sums


def _encoded_labels(labels: np.ndarray, classes: np.ndarray, *, name: str) -> np.ndarray:
    class_to_index = {class_label: class_index for class_index, class_label in enumerate(classes.tolist())}
    encoded = []
    for label in labels.tolist():
        if label not in class_to_index:
            raise ValueError(f"{name} contains class {label!r}, which is not present in the requested classes.")
        encoded.append(class_to_index[label])
    return np.asarray(encoded, dtype=np.int64)


def _balanced_class_weights(encoded: np.ndarray, n_classes: int, device):
    torch = _torch()
    counts = np.bincount(encoded.astype(int, copy=False), minlength=n_classes).astype(np.float32)
    weights = encoded.shape[0] / np.maximum(counts, 1.0) / float(n_classes)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _softmax_entropy(probabilities) -> Any:
    torch = _torch()
    return -(probabilities * torch.log(torch.clamp(probabilities, min=1e-8))).sum(dim=1).mean()


_TORCH_MODULE_BASE = object if _TORCH is None else _TORCH.nn.Module


class _LoRALinear(_TORCH_MODULE_BASE):  # type: ignore[misc, valid-type]
    def __init__(self, in_features: int, out_features: int, *, rank: int, alpha: float, bias: bool = True):
        torch = _torch()
        super().__init__()
        self.base = torch.nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(max(1, rank))
        self.lora_a = torch.nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_b = torch.nn.Parameter(torch.zeros(out_features, self.rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        torch = _torch()
        torch.nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_b)

    def forward(self, features):
        return self.base(features) + ((features @ self.lora_a.t()) @ self.lora_b.t()) * self.scaling


class _LoRAMLP(_TORCH_MODULE_BASE):  # type: ignore[misc, valid-type]
    def __init__(self, input_dim: int, hidden_units: int, n_classes: int, *, rank: int, alpha: float, dropout: float):
        torch = _torch()
        super().__init__()
        self.hidden = _LoRALinear(input_dim, hidden_units, rank=rank, alpha=alpha)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(float(dropout))
        self.head = _LoRALinear(hidden_units, n_classes, rank=rank, alpha=alpha)

    def forward(self, features):
        hidden = self.dropout(self.activation(self.hidden(features)))
        return self.head(hidden)


def _set_source_pretrain_trainable(model: _LoRAMLP) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" not in name)


def _is_adapter_parameter(name: str, *, adapt_head: str) -> bool:
    if ".lora_" in name:
        return True
    if adapt_head == "bias" and name == "head.base.bias":
        return True
    if adapt_head == "full" and name.startswith("head.base."):
        return True
    return False


def _set_adapter_trainable(model: _LoRAMLP, *, adapt_head: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_adapter_parameter(name, adapt_head=adapt_head))


def _trainable_parameters(model: _LoRAMLP) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _adapter_state(model: _LoRAMLP, *, adapt_head: str) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters() if _is_adapter_parameter(name, adapt_head=adapt_head)}


def _load_adapter_state(model: _LoRAMLP, state: dict[str, Any]) -> None:
    for name, parameter in model.named_parameters():
        if name in state:
            parameter.data.copy_(state[name].to(parameter.device))


def _iter_minibatches(n_rows: int, *, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _supervised_pretrain(
    model: _LoRAMLP,
    features,
    labels,
    *,
    n_classes: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    class_weight: str | None,
    rng: np.random.Generator,
) -> int:
    torch = _torch()
    _set_source_pretrain_trainable(model)
    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=float(learning_rate), weight_decay=float(weight_decay))
    weights = _balanced_class_weights(labels.detach().cpu().numpy(), n_classes, features.device) if class_weight == "balanced" else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    epochs_run = 0
    for epoch in range(int(max_epochs)):
        epochs_run = epoch + 1
        model.train()
        for batch in _iter_minibatches(labels.shape[0], batch_size=int(batch_size), rng=rng):
            batch_tensor = torch.as_tensor(batch, dtype=torch.long, device=features.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[batch_tensor]), labels[batch_tensor])
            loss.backward()
            optimizer.step()
    return epochs_run


def _target_adaptation_loss(
    model: _LoRAMLP,
    calibration_features,
    calibration_labels,
    unlabeled_features,
    *,
    loss_fn,
    entropy_loss_weight: float,
    consistency_loss_weight: float,
    consistency_noise_std: float,
):
    torch = _torch()
    supervised_loss = loss_fn(model(calibration_features), calibration_labels)
    loss = supervised_loss
    entropy_loss = torch.as_tensor(0.0, device=calibration_features.device)
    consistency_loss = torch.as_tensor(0.0, device=calibration_features.device)
    if unlabeled_features is not None and unlabeled_features.shape[0] > 0:
        probabilities = torch.softmax(model(unlabeled_features), dim=1)
        if entropy_loss_weight > 0.0:
            entropy_loss = _softmax_entropy(probabilities)
            loss = loss + float(entropy_loss_weight) * entropy_loss
        if consistency_loss_weight > 0.0:
            noise_std = float(consistency_noise_std)
            noisy = unlabeled_features + noise_std * torch.randn_like(unlabeled_features)
            noisy_probabilities = torch.softmax(model(noisy), dim=1)
            consistency_loss = torch.mean((probabilities.detach() - noisy_probabilities) ** 2)
            loss = loss + float(consistency_loss_weight) * consistency_loss
    return loss, supervised_loss.detach(), entropy_loss.detach(), consistency_loss.detach()


def _adapt_on_target(
    model: _LoRAMLP,
    calibration_features,
    calibration_labels,
    unlabeled_features,
    *,
    n_classes: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    adapt_head: str,
    class_weight: str | None,
    entropy_loss_weight: float,
    consistency_loss_weight: float,
    consistency_noise_std: float,
) -> dict[str, float | int]:
    torch = _torch()
    _set_adapter_trainable(model, adapt_head=adapt_head)
    trainable = _trainable_parameters(model)
    if not trainable:
        raise ValueError("LoRA few-shot adaptation has no trainable parameters; use adapt_head != 'none' or lora_rank >= 1.")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(weight_decay))
    weights = _balanced_class_weights(calibration_labels.detach().cpu().numpy(), n_classes, calibration_features.device) if class_weight == "balanced" else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    last = {
        "target_adaptation_loss": np.nan,
        "target_adaptation_supervised_loss": np.nan,
        "target_adaptation_entropy_loss": np.nan,
        "target_adaptation_consistency_loss": np.nan,
    }
    for _step in range(int(steps)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, supervised_loss, entropy_loss, consistency_loss = _target_adaptation_loss(
            model,
            calibration_features,
            calibration_labels,
            unlabeled_features,
            loss_fn=loss_fn,
            entropy_loss_weight=entropy_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            consistency_noise_std=consistency_noise_std,
        )
        loss.backward()
        optimizer.step()
        last = {
            "target_adaptation_loss": float(loss.detach().cpu()),
            "target_adaptation_supervised_loss": float(supervised_loss.cpu()),
            "target_adaptation_entropy_loss": float(entropy_loss.cpu()),
            "target_adaptation_consistency_loss": float(consistency_loss.cpu()),
        }
    last["target_adaptation_steps_run"] = int(steps)
    return last


def _source_episode_indices(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    support_per_class: int,
    query_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] == 0:
        return None
    for group in rng.permutation(unique_groups):
        group_indices = np.flatnonzero(groups == group)
        support: list[int] = []
        query: list[int] = []
        ok = True
        for class_index in range(n_classes):
            class_indices = group_indices[labels[group_indices] == class_index]
            needed = int(support_per_class) + int(query_per_class)
            if class_indices.shape[0] < needed:
                ok = False
                break
            chosen = rng.choice(class_indices, size=needed, replace=False)
            support.extend(chosen[:support_per_class].tolist())
            query.extend(chosen[support_per_class:].tolist())
        if ok:
            return np.asarray(support, dtype=int), np.asarray(query, dtype=int)
    return None


def _reptile_meta_train(
    model: _LoRAMLP,
    features,
    labels,
    groups: np.ndarray | None,
    *,
    n_classes: int,
    meta_epochs: int,
    inner_steps: int,
    support_per_class: int,
    query_per_class: int,
    adapter_learning_rate: float,
    weight_decay: float,
    adapt_head: str,
    class_weight: str | None,
    meta_step_size: float,
    entropy_loss_weight: float,
    consistency_loss_weight: float,
    consistency_noise_std: float,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    torch = _torch()
    if groups is None or int(meta_epochs) <= 0:
        return {"meta_learning_enabled": False, "meta_episodes": 0}
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if groups.shape[0] != labels.shape[0]:
        raise ValueError("source_groups must contain one group label per source row.")
    if np.unique(groups).shape[0] < 2:
        return {"meta_learning_enabled": False, "meta_episodes": 0}

    _set_adapter_trainable(model, adapt_head=adapt_head)
    episodes = 0
    query_losses: list[float] = []
    for _epoch in range(int(meta_epochs)):
        episode = _source_episode_indices(
            labels=labels.detach().cpu().numpy(),
            groups=groups,
            n_classes=n_classes,
            support_per_class=int(support_per_class),
            query_per_class=int(query_per_class),
            rng=rng,
        )
        if episode is None:
            continue
        support_idx, query_idx = episode
        clone = copy.deepcopy(model)
        before = _adapter_state(model, adapt_head=adapt_head)
        _load_adapter_state(clone, before)
        _adapt_on_target(
            clone,
            features[torch.as_tensor(support_idx, dtype=torch.long, device=features.device)],
            labels[torch.as_tensor(support_idx, dtype=torch.long, device=features.device)],
            features[torch.as_tensor(query_idx, dtype=torch.long, device=features.device)],
            n_classes=n_classes,
            steps=int(inner_steps),
            learning_rate=float(adapter_learning_rate),
            weight_decay=float(weight_decay),
            adapt_head=adapt_head,
            class_weight=class_weight,
            entropy_loss_weight=float(entropy_loss_weight),
            consistency_loss_weight=float(consistency_loss_weight),
            consistency_noise_std=float(consistency_noise_std),
        )
        clone.eval()
        with torch.no_grad():
            query_logits = clone(features[torch.as_tensor(query_idx, dtype=torch.long, device=features.device)])
            query_loss = torch.nn.functional.cross_entropy(query_logits, labels[torch.as_tensor(query_idx, dtype=torch.long, device=features.device)])
            query_losses.append(float(query_loss.detach().cpu()))
        clone_state = _adapter_state(clone, adapt_head=adapt_head)
        for name, parameter in model.named_parameters():
            if name in clone_state:
                parameter.data.add_(float(meta_step_size) * (clone_state[name].to(parameter.device) - parameter.data))
        episodes += 1
    return {
        "meta_learning_enabled": episodes > 0,
        "meta_episodes": int(episodes),
        "meta_mean_query_loss": float(np.mean(query_losses)) if query_losses else np.nan,
    }


class SemiSupervisedLoRAFewShotClassifier(ClassifierMixin, BaseEstimator):
    """Source-meta-trained LoRA adapter for Category-3 target calibration.

    ``fit`` uses source labels, labeled target calibration rows, and optionally
    unlabeled target rows.  It never accepts target evaluation labels.  The
    source stage trains a base MLP; when ``source_groups`` are supplied, a
    Reptile-style episodic pass meta-learns adapter/head initialisation by
    treating source subjects as pseudo-target tasks.  The target stage freezes the
    base network and adapts only LoRA adapter parameters plus the requested head
    subset on the labeled calibration rows and optional unlabeled target rows.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        dropout: float = 0.1,
        source_pretrain_epochs: int = 80,
        meta_epochs: int = 20,
        meta_inner_steps: int = 5,
        meta_support_per_class: int = 1,
        meta_query_per_class: int = 1,
        meta_step_size: float = 0.2,
        target_adaptation_steps: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        adapter_learning_rate: float = 5e-3,
        weight_decay: float = 1e-4,
        entropy_loss_weight: float = 0.02,
        consistency_loss_weight: float = 0.0,
        consistency_noise_std: float = 0.05,
        adapt_head: str = "bias",
        class_weight: str | None = "balanced",
        random_state: int | None = 13,
        device: str = "auto",
    ):
        self.hidden_units = hidden_units
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.dropout = dropout
        self.source_pretrain_epochs = source_pretrain_epochs
        self.meta_epochs = meta_epochs
        self.meta_inner_steps = meta_inner_steps
        self.meta_support_per_class = meta_support_per_class
        self.meta_query_per_class = meta_query_per_class
        self.meta_step_size = meta_step_size
        self.target_adaptation_steps = target_adaptation_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.adapter_learning_rate = adapter_learning_rate
        self.weight_decay = weight_decay
        self.entropy_loss_weight = entropy_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_noise_std = consistency_noise_std
        self.adapt_head = adapt_head
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device

    def _resolve_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def fit(
        self,
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_labels: Sequence[Any] | np.ndarray,
        *,
        target_calibration_features: Sequence[Sequence[float]] | np.ndarray,
        target_calibration_labels: Sequence[Any] | np.ndarray,
        target_unlabeled_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        source_groups: Sequence[Any] | np.ndarray | None = None,
        classes: Sequence[Any] | np.ndarray | None = None,
    ):
        torch = _torch()
        source_matrix = _as_feature_matrix(source_features, name="source_features")
        calibration_matrix = _as_feature_matrix(target_calibration_features, name="target_calibration_features")
        if source_matrix.shape[1] != calibration_matrix.shape[1]:
            raise ValueError("source_features and target_calibration_features must have the same feature width.")
        unlabeled_matrix = _as_optional_feature_matrix(target_unlabeled_features, name="target_unlabeled_features", n_features=source_matrix.shape[1])

        source_label_vector = _as_1d_object_array(source_labels, name="source_labels")
        calibration_label_vector = _as_1d_object_array(target_calibration_labels, name="target_calibration_labels")
        if source_label_vector.shape[0] != source_matrix.shape[0]:
            raise ValueError("source_features and source_labels must have the same row count.")
        if calibration_label_vector.shape[0] != calibration_matrix.shape[0]:
            raise ValueError("target_calibration_features and target_calibration_labels must have the same row count.")

        hidden_units = _positive_int(self.hidden_units, name="hidden_units")
        lora_rank = _positive_int(self.lora_rank, name="lora_rank")
        lora_alpha = _positive_float(self.lora_alpha, name="lora_alpha")
        source_pretrain_epochs = _nonnegative_int(self.source_pretrain_epochs, name="source_pretrain_epochs")
        meta_epochs = _nonnegative_int(self.meta_epochs, name="meta_epochs")
        meta_inner_steps = _positive_int(self.meta_inner_steps, name="meta_inner_steps")
        meta_support_per_class = _positive_int(self.meta_support_per_class, name="meta_support_per_class")
        meta_query_per_class = _positive_int(self.meta_query_per_class, name="meta_query_per_class")
        target_adaptation_steps = _nonnegative_int(self.target_adaptation_steps, name="target_adaptation_steps")
        batch_size = _positive_int(self.batch_size, name="batch_size")
        learning_rate = _positive_float(self.learning_rate, name="learning_rate")
        adapter_learning_rate = _positive_float(self.adapter_learning_rate, name="adapter_learning_rate")
        weight_decay = _nonnegative_float(self.weight_decay, name="weight_decay")
        entropy_loss_weight = _nonnegative_float(self.entropy_loss_weight, name="entropy_loss_weight")
        consistency_loss_weight = _nonnegative_float(self.consistency_loss_weight, name="consistency_loss_weight")
        consistency_noise_std = _nonnegative_float(self.consistency_noise_std, name="consistency_noise_std")
        dropout = _bounded_float(self.dropout, name="dropout", lower=0.0, upper=1.0)
        meta_step_size = _bounded_float(self.meta_step_size, name="meta_step_size", lower=0.0, upper=1.0)
        adapt_head = _normalize_adapt_head(self.adapt_head)
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'.")

        if classes is None:
            class_order = np.unique(np.concatenate([source_label_vector, calibration_label_vector]).astype(object))
        else:
            class_order = np.asarray(classes, dtype=object).reshape(-1)
        if class_order.shape[0] < 2:
            raise ValueError("Semi-supervised LoRA few-shot calibration requires at least two classes.")
        source_encoded = _encoded_labels(source_label_vector, class_order, name="source_labels")
        calibration_encoded = _encoded_labels(calibration_label_vector, class_order, name="target_calibration_labels")

        if self.random_state is not None:
            random_state = _nonnegative_int(self.random_state, name="random_state")
        else:
            random_state = None
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
        rng = np.random.default_rng(random_state)
        device = self._resolve_device()

        scaler_rows = [source_matrix, calibration_matrix]
        if unlabeled_matrix.shape[0] > 0:
            scaler_rows.append(unlabeled_matrix)
        scaler_matrix = np.vstack(scaler_rows)
        feature_mean = scaler_matrix.mean(axis=0, keepdims=True)
        feature_scale = scaler_matrix.std(axis=0, keepdims=True)
        feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
        source_scaled = (source_matrix - feature_mean) / feature_scale
        calibration_scaled = (calibration_matrix - feature_mean) / feature_scale
        unlabeled_scaled = (unlabeled_matrix - feature_mean) / feature_scale if unlabeled_matrix.shape[0] > 0 else unlabeled_matrix

        source_tensor = torch.as_tensor(source_scaled, dtype=torch.float32, device=device)
        source_label_tensor = torch.as_tensor(source_encoded, dtype=torch.long, device=device)
        calibration_tensor = torch.as_tensor(calibration_scaled, dtype=torch.float32, device=device)
        calibration_label_tensor = torch.as_tensor(calibration_encoded, dtype=torch.long, device=device)
        unlabeled_tensor = torch.as_tensor(unlabeled_scaled, dtype=torch.float32, device=device) if unlabeled_scaled.shape[0] > 0 else None

        model = _LoRAMLP(
            input_dim=source_matrix.shape[1],
            hidden_units=hidden_units,
            n_classes=class_order.shape[0],
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=dropout,
        ).to(device)
        pretrain_epochs_run = _supervised_pretrain(
            model,
            source_tensor,
            source_label_tensor,
            n_classes=class_order.shape[0],
            max_epochs=source_pretrain_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weight=self.class_weight,
            rng=rng,
        )

        meta_info = _reptile_meta_train(
            model,
            source_tensor,
            source_label_tensor,
            None if source_groups is None else np.asarray(source_groups, dtype=object).reshape(-1),
            n_classes=class_order.shape[0],
            meta_epochs=meta_epochs,
            inner_steps=meta_inner_steps,
            support_per_class=meta_support_per_class,
            query_per_class=meta_query_per_class,
            adapter_learning_rate=adapter_learning_rate,
            weight_decay=weight_decay,
            adapt_head=adapt_head,
            class_weight=self.class_weight,
            meta_step_size=meta_step_size,
            entropy_loss_weight=entropy_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            consistency_noise_std=consistency_noise_std,
            rng=rng,
        )
        target_info = _adapt_on_target(
            model,
            calibration_tensor,
            calibration_label_tensor,
            unlabeled_tensor,
            n_classes=class_order.shape[0],
            steps=target_adaptation_steps,
            learning_rate=adapter_learning_rate,
            weight_decay=weight_decay,
            adapt_head=adapt_head,
            class_weight=self.class_weight,
            entropy_loss_weight=entropy_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            consistency_noise_std=consistency_noise_std,
        )

        model.eval()
        self.model_ = model
        self.classes_ = class_order
        self.device_ = device
        self.n_features_in_ = source_matrix.shape[1]
        self.feature_mean_ = feature_mean.astype(float, copy=True)
        self.feature_scale_ = feature_scale.astype(float, copy=True)
        self.source_rows_ = int(source_matrix.shape[0])
        self.target_calibration_rows_ = int(calibration_matrix.shape[0])
        self.target_unlabeled_rows_ = int(unlabeled_matrix.shape[0])
        self.source_pretrain_epochs_run_ = int(pretrain_epochs_run)
        self.meta_info_ = dict(meta_info)
        self.target_adaptation_info_ = dict(target_info)
        self.metadata_ = self._metadata()
        return self

    def _metadata(self) -> dict[str, Any]:
        meta_episodes = int(getattr(self, "meta_info_", {}).get("meta_episodes", 0))
        metadata = {
            "few_shot_target_calibration": True,
            "few_shot_protocol": SEMI_SUPERVISED_LORA_FEW_SHOT_PROTOCOL,
            "few_shot_protocol_category": SEMI_SUPERVISED_LORA_FEW_SHOT_CATEGORY,
            "semi_supervised_lora_few_shot": True,
            "semi_supervised_lora_meta_algorithm": SEMI_SUPERVISED_LORA_META_ALGORITHM,
            "semi_supervised_lora_uses_source_features": True,
            "semi_supervised_lora_uses_source_labels": True,
            "semi_supervised_lora_uses_target_calibration_features": True,
            "semi_supervised_lora_uses_target_calibration_labels": True,
            "semi_supervised_lora_uses_unlabeled_target_features": int(getattr(self, "target_unlabeled_rows_", 0)) > 0,
            "semi_supervised_lora_uses_target_evaluation_labels": False,
            "semi_supervised_lora_valid_for_strict_source_only": False,
            "semi_supervised_lora_valid_for_unlabeled_target_adaptation": False,
            "semi_supervised_lora_valid_for_zero_calibration": False,
            "semi_supervised_lora_valid_for_benchmark": False,
            "semi_supervised_lora_rank": int(self.lora_rank),
            "semi_supervised_lora_alpha": float(self.lora_alpha),
            "semi_supervised_lora_adapt_head": _normalize_adapt_head(self.adapt_head),
            "semi_supervised_lora_source_pretrain_epochs_run": int(getattr(self, "source_pretrain_epochs_run_", 0)),
            "semi_supervised_lora_meta_learning_enabled": meta_episodes > 0,
            "semi_supervised_lora_meta_episodes": meta_episodes,
            "semi_supervised_lora_meta_algorithm_used": SEMI_SUPERVISED_LORA_META_ALGORITHM if meta_episodes > 0 else "none",
            "semi_supervised_lora_target_adaptation_steps": int(getattr(self, "target_adaptation_info_", {}).get("target_adaptation_steps_run", 0)),
            "semi_supervised_lora_entropy_loss_weight": float(self.entropy_loss_weight),
            "semi_supervised_lora_consistency_loss_weight": float(self.consistency_loss_weight),
            "semi_supervised_lora_source_rows": int(getattr(self, "source_rows_", 0)),
            "semi_supervised_lora_target_calibration_rows": int(getattr(self, "target_calibration_rows_", 0)),
            "semi_supervised_lora_unlabeled_target_rows": int(getattr(self, "target_unlabeled_rows_", 0)),
        }
        metadata.update({f"semi_supervised_lora_{key}": value for key, value in getattr(self, "meta_info_", {}).items()})
        metadata.update({f"semi_supervised_lora_{key}": value for key, value in getattr(self, "target_adaptation_info_", {}).items()})
        return metadata

    def _transform_features(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "feature_mean_") or not hasattr(self, "feature_scale_"):
            raise RuntimeError("SemiSupervisedLoRAFewShotClassifier must be fitted before prediction.")
        matrix = _as_feature_matrix(features, name="features")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(f"features must have {self.n_features_in_} columns; got {matrix.shape[1]}.")
        return (matrix - self.feature_mean_) / self.feature_scale_

    def _logits(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("SemiSupervisedLoRAFewShotClassifier must be fitted before prediction.")
        torch = _torch()
        tensor = torch.as_tensor(self._transform_features(features), dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(tensor)
        return logits.detach().cpu().numpy()

    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        logits = self._logits(features)
        if logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        return logits

    def predict_proba(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        torch = _torch()
        logits = torch.as_tensor(self._logits(features), dtype=torch.float32)
        return _normalize_probability_rows(torch.softmax(logits, dim=1).detach().cpu().numpy())

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


def fit_semi_supervised_lora_few_shot_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    source_groups: Sequence[Any] | np.ndarray | None = None,
    classes: Sequence[Any] | np.ndarray | None = None,
    split: FewShotTargetCalibrationSplit | None = None,
    per_class: int | str = 1,
    seed: int | str = 13,
    context: Sequence[Hashable] = (),
    min_evaluation_per_class: int | str = 1,
    extra_unlabeled_target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    use_evaluation_features_unlabeled: bool = True,
    **model_kwargs: Any,
) -> SemiSupervisedLoRAFewShotResult:
    """Fit a semi-supervised LoRA few-shot model and score disjoint target rows.

    The target labels are used only for selecting and labeling the calibration
    subset.  Evaluation labels are not passed into the estimator; evaluation
    feature rows may optionally be used as unlabeled target data for
    transductive/semi-supervised losses and are reported in metadata.
    """

    source_matrix = _as_feature_matrix(source_features, name="source_features")
    target_matrix = _as_feature_matrix(target_features, name="target_features")
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError("source_features and target_features must have the same feature width.")
    source_label_vector = _as_1d_object_array(source_labels, name="source_labels")
    target_label_vector = _as_1d_object_array(target_labels, name="target_labels")
    if source_label_vector.shape[0] != source_matrix.shape[0]:
        raise ValueError("source_features and source_labels must have the same row count.")
    if target_label_vector.shape[0] != target_matrix.shape[0]:
        raise ValueError("target_features and target_labels must have the same row count.")

    per_class_count = _positive_int(per_class, name="few_shot_target_calibration_per_class")
    seed_value = _nonnegative_int(seed, name="few_shot_target_calibration_seed")
    if split is None:
        split = select_few_shot_target_calibration_split(
            target_label_vector,
            per_class=per_class_count,
            seed=seed_value,
            context=context,
            min_evaluation_per_class=min_evaluation_per_class,
        )
    calibration_indices = np.asarray(split.calibration_indices, dtype=int).reshape(-1)
    evaluation_indices = np.asarray(split.evaluation_indices, dtype=int).reshape(-1)
    if calibration_indices.size == 0:
        raise ValueError("semi-supervised LoRA few-shot calibration selected no calibration rows.")
    if evaluation_indices.size == 0:
        raise ValueError("semi-supervised LoRA few-shot calibration selected no evaluation rows.")
    for name, indices in {"calibration_indices": calibration_indices, "evaluation_indices": evaluation_indices}.items():
        if np.any(indices < 0) or np.any(indices >= target_matrix.shape[0]):
            raise ValueError(f"{name} contains an out-of-range target row index.")
    if np.intersect1d(calibration_indices, evaluation_indices).size:
        raise ValueError("few-shot calibration and evaluation indices must be disjoint.")

    unlabeled_blocks = []
    if use_evaluation_features_unlabeled:
        unlabeled_blocks.append(target_matrix[evaluation_indices])
    if extra_unlabeled_target_features is not None:
        extra = _as_feature_matrix(extra_unlabeled_target_features, name="extra_unlabeled_target_features", allow_empty=True)
        if extra.shape[1] != target_matrix.shape[1]:
            raise ValueError("extra_unlabeled_target_features must have the same feature width as target_features.")
        if extra.shape[0] > 0:
            unlabeled_blocks.append(extra)
    target_unlabeled = np.vstack(unlabeled_blocks) if unlabeled_blocks else None

    class_order = np.asarray(classes, dtype=object).reshape(-1) if classes is not None else np.unique(np.concatenate([source_label_vector, target_label_vector[calibration_indices]]).astype(object))
    model = SemiSupervisedLoRAFewShotClassifier(random_state=seed_value, **model_kwargs)
    model.fit(
        source_matrix,
        source_label_vector,
        target_calibration_features=target_matrix[calibration_indices],
        target_calibration_labels=target_label_vector[calibration_indices],
        target_unlabeled_features=target_unlabeled,
        source_groups=source_groups,
        classes=class_order,
    )
    probabilities = model.predict_proba(target_matrix[evaluation_indices])
    metadata = dict(model.metadata_)
    metadata.update(
        {
            "few_shot_protocol": SEMI_SUPERVISED_LORA_FEW_SHOT_PROTOCOL,
            "few_shot_protocol_category": SEMI_SUPERVISED_LORA_FEW_SHOT_CATEGORY,
            "few_shot_target_calibration": True,
            "few_shot_target_calibration_per_class": int(per_class_count),
            "few_shot_target_calibration_seed": int(seed_value),
            "few_shot_n_source_rows": int(source_matrix.shape[0]),
            "few_shot_n_target_rows": int(target_matrix.shape[0]),
            "few_shot_n_target_calibration_rows": int(calibration_indices.size),
            "few_shot_n_target_evaluation_rows": int(evaluation_indices.size),
            "semi_supervised_lora_transductive_evaluation_features": bool(use_evaluation_features_unlabeled),
            "semi_supervised_lora_extra_unlabeled_target_rows": 0 if extra_unlabeled_target_features is None else int(np.asarray(extra_unlabeled_target_features).shape[0]),
        }
    )
    return SemiSupervisedLoRAFewShotResult(
        model=model,
        probabilities=probabilities,
        evaluation_indices=evaluation_indices.copy(),
        calibration_indices=calibration_indices.copy(),
        metadata=metadata,
    )
