"""Progressive target adaptation for one-label MEG windows.

This is the window-level counterpart of the sequence-event model. It keeps the
same central idea: a source backbone is trained without target data, then an
identity-initialized low-rank residual and the head are adapted before broader
parts of the network are allowed to move.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from neureptrace.decoding._progressive_sequence_core import _torch


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def _nonnegative_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return parsed


def _as_float_matrix(values: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _as_label_vector(values: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    return result


def _WindowAdapterModule(
    *,
    input_dim: int,
    n_classes: int,
    hidden_units: int,
    num_layers: int,
    adapter_rank: int,
    adapter_alpha: float,
    dropout: float,
):
    torch = _torch()

    class ResidualBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden_units)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden_units, hidden_units * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_units * 2, hidden_units),
                torch.nn.Dropout(dropout),
            )

        def forward(self, values):
            return values + self.network(self.norm(values))

    class Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = torch.nn.LayerNorm(input_dim)
            self.base_projection = torch.nn.Linear(input_dim, hidden_units)
            self.adapter_down = torch.nn.Linear(input_dim, adapter_rank, bias=False)
            self.adapter_up = torch.nn.Linear(adapter_rank, hidden_units, bias=False)
            self.adapter_scale = float(adapter_alpha) / float(adapter_rank)
            self.hidden_blocks = torch.nn.ModuleList(ResidualBlock() for _ in range(num_layers))
            self.final_norm = torch.nn.LayerNorm(hidden_units)
            self.class_head = torch.nn.Linear(hidden_units, n_classes)
            torch.nn.init.zeros_(self.adapter_up.weight)

        def forward(self, values, *, use_adapter: bool):
            normalized = self.input_norm(values)
            hidden = self.base_projection(normalized)
            if use_adapter:
                hidden = hidden + self.adapter_scale * self.adapter_up(self.adapter_down(normalized))
            hidden = torch.nn.functional.gelu(hidden)
            for block in self.hidden_blocks:
                hidden = block(hidden)
            return self.class_head(self.final_norm(hidden))

    return Module()


class TorchProgressiveWindowClassifier:
    """Source-pretrained MLP with low-rank and progressive target stages."""

    def __init__(
        self,
        *,
        hidden_units: int = 128,
        num_layers: int = 2,
        adapter_rank: int = 8,
        adapter_alpha: float = 8.0,
        source_epochs: int = 12,
        source_learning_rate: float = 1e-3,
        adapter_steps: int = 80,
        last_block_steps: int = 60,
        full_finetune_steps: int = 60,
        adaptation_learning_rate: float = 2e-3,
        full_finetune_learning_rate: float = 2e-4,
        batch_size: int = 1024,
        weight_decay: float = 1e-4,
        dropout: float = 0.15,
        feature_noise_std: float = 0.02,
        source_replay_weight: float = 0.1,
        l2sp_weight: float = 1e-4,
        min_trials_for_last_block: int = 8,
        min_trials_for_full_finetune: int = 12,
        random_state: int = 0,
        device: str = "auto",
    ) -> None:
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.source_epochs = source_epochs
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
        self.source_replay_weight = source_replay_weight
        self.l2sp_weight = l2sp_weight
        self.min_trials_for_last_block = min_trials_for_last_block
        self.min_trials_for_full_finetune = min_trials_for_full_finetune
        self.random_state = random_state
        self.device = device

    def _resolved_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _seed_torch(self) -> None:
        torch = _torch()
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

    def _build_model(self) -> None:
        hidden_units = _positive_int(self.hidden_units, "hidden_units")
        self.model_ = _WindowAdapterModule(
            input_dim=self.n_features_in_,
            n_classes=self.n_classes_,
            hidden_units=hidden_units,
            num_layers=_positive_int(self.num_layers, "num_layers"),
            adapter_rank=_positive_int(self.adapter_rank, "adapter_rank"),
            adapter_alpha=float(self.adapter_alpha),
            dropout=float(self.dropout),
        ).to(self.device_)

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        mapping = {label: position for position, label in enumerate(self.classes_.tolist())}
        try:
            return np.asarray([mapping[label] for label in labels.tolist()], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Target label {error.args[0]!r} is absent from source classes") from error

    def _iter_epoch_batches(self, n_rows: int, rng: np.random.Generator):
        batch_size = _positive_int(self.batch_size, "batch_size")
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, batch_size):
            yield order[start : start + batch_size]

    def _augment(self, tensor):
        torch = _torch()
        std = float(self.feature_noise_std)
        if std <= 0.0:
            return tensor
        return tensor + std * torch.randn_like(tensor)

    def fit_source(
        self,
        source_features: Sequence | np.ndarray,
        source_labels: Sequence | np.ndarray,
    ) -> "TorchProgressiveWindowClassifier":
        torch = _torch()
        x = _as_float_matrix(source_features, name="source_features")
        labels = _as_label_vector(source_labels, name="source_labels")
        if labels.shape[0] != x.shape[0]:
            raise ValueError("source_features and source_labels must contain the same number of rows")
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("Source fitting requires at least two classes")
        self.n_features_in_ = int(x.shape[1])
        self.n_classes_ = int(self.classes_.size)
        self.device_ = self._resolved_device()
        self._seed_torch()
        self._build_model()
        for name, parameter in self.model_.named_parameters():
            parameter.requires_grad_(not name.startswith("adapter_"))
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model_.parameters() if parameter.requires_grad),
            lr=float(self.source_learning_rate),
            weight_decay=float(self.weight_decay),
        )
        rng = np.random.default_rng(int(self.random_state))
        self.model_.train()
        for _epoch in range(_positive_int(self.source_epochs, "source_epochs")):
            for batch_indices in self._iter_epoch_batches(x.shape[0], rng):
                batch_x = torch.as_tensor(x[batch_indices], dtype=torch.float32, device=self.device_)
                batch_y = torch.as_tensor(encoded[batch_indices], dtype=torch.long, device=self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(self._augment(batch_x), use_adapter=False)
                loss = torch.nn.functional.cross_entropy(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=5.0)
                optimizer.step()
        self.source_features_ = x
        self.source_encoded_labels_ = encoded.astype(np.int64, copy=False)
        self.source_state_ = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
        self.model_.eval()
        return self

    def clone_source(self) -> "TorchProgressiveWindowClassifier":
        if not hasattr(self, "source_state_"):
            raise RuntimeError("fit_source must be called before clone_source")
        clone = copy.copy(self)
        clone._build_model()
        clone.model_.load_state_dict(self.source_state_)
        clone.source_state_ = self.source_state_
        clone.source_features_ = self.source_features_
        clone.source_encoded_labels_ = self.source_encoded_labels_
        clone.adaptation_history_ = []
        clone.model_.eval()
        return clone

    def _set_stage_trainable(self, stage: str) -> tuple[str, ...]:
        for parameter in self.model_.parameters():
            parameter.requires_grad_(False)
        prefixes = ["adapter_", "class_head", "final_norm"]
        if stage in {"last_block", "full"}:
            prefixes.append(f"hidden_blocks.{len(self.model_.hidden_blocks) - 1}")
        if stage == "full":
            for parameter in self.model_.parameters():
                parameter.requires_grad_(True)
        else:
            for name, parameter in self.model_.named_parameters():
                if any(name.startswith(prefix) for prefix in prefixes):
                    parameter.requires_grad_(True)
        return tuple(name for name, parameter in self.model_.named_parameters() if parameter.requires_grad)

    def _l2sp_penalty(self):
        torch = _torch()
        penalty = torch.zeros((), dtype=torch.float32, device=self.device_)
        for name, parameter in self.model_.named_parameters():
            if parameter.requires_grad and name in self.source_state_:
                reference = self.source_state_[name].to(self.device_)
                penalty = penalty + torch.sum((parameter - reference) ** 2)
        return penalty

    def _adapt_stage(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        *,
        stage: str,
        steps: int,
        learning_rate: float,
    ) -> None:
        torch = _torch()
        trainable_names = self._set_stage_trainable(stage)
        parameters = [parameter for parameter in self.model_.parameters() if parameter.requires_grad]
        if not parameters or steps <= 0:
            return
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(self.weight_decay))
        rng = np.random.default_rng(int(self.random_state) + {"adapter": 11, "last_block": 23, "full": 37}[stage])
        batch_size = _positive_int(self.batch_size, "batch_size")
        for _step in range(steps):
            target_indices = rng.choice(
                target_x.shape[0],
                size=min(batch_size, target_x.shape[0]),
                replace=target_x.shape[0] < batch_size,
            )
            batch_x = torch.as_tensor(target_x[target_indices], dtype=torch.float32, device=self.device_)
            batch_y = torch.as_tensor(target_y[target_indices], dtype=torch.long, device=self.device_)
            optimizer.zero_grad(set_to_none=True)
            logits = self.model_(self._augment(batch_x), use_adapter=True)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            if float(self.source_replay_weight) > 0.0:
                source_indices = rng.choice(
                    self.source_features_.shape[0],
                    size=min(batch_size, self.source_features_.shape[0]),
                    replace=self.source_features_.shape[0] < batch_size,
                )
                source_x = torch.as_tensor(self.source_features_[source_indices], dtype=torch.float32, device=self.device_)
                source_y = torch.as_tensor(self.source_encoded_labels_[source_indices], dtype=torch.long, device=self.device_)
                source_logits = self.model_(self._augment(source_x), use_adapter=False)
                loss = loss + float(self.source_replay_weight) * torch.nn.functional.cross_entropy(source_logits, source_y)
            if float(self.l2sp_weight) > 0.0:
                loss = loss + float(self.l2sp_weight) * self._l2sp_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
        self.adaptation_history_.append(
            {
                "stage": stage,
                "steps": int(steps),
                "trainable_parameter_names": trainable_names,
            }
        )

    def adapt_target(
        self,
        target_calibration_features: Sequence | np.ndarray,
        target_calibration_labels: Sequence | np.ndarray,
        *,
        n_calibration_trials: int,
        mode: str = "progressive_full",
    ) -> "TorchProgressiveWindowClassifier":
        if not hasattr(self, "source_state_"):
            raise RuntimeError("fit_source must be called before adapt_target")
        if mode not in {"adapter_only", "progressive_full"}:
            raise ValueError("mode must be 'adapter_only' or 'progressive_full'")
        x = _as_float_matrix(target_calibration_features, name="target_calibration_features")
        labels = _as_label_vector(target_calibration_labels, name="target_calibration_labels")
        if x.shape[1] != self.n_features_in_ or labels.shape[0] != x.shape[0]:
            raise ValueError("Target calibration arrays do not match the fitted source dimensions")
        encoded = self._encode_labels(labels)
        self.model_.load_state_dict(self.source_state_)
        self.adaptation_history_ = []
        stages: list[tuple[str, int, float]] = [("adapter", _nonnegative_int(self.adapter_steps, "adapter_steps"), float(self.adaptation_learning_rate))]
        if mode == "progressive_full" and n_calibration_trials >= int(self.min_trials_for_last_block):
            stages.append(("last_block", _nonnegative_int(self.last_block_steps, "last_block_steps"), float(self.adaptation_learning_rate)))
        if mode == "progressive_full" and n_calibration_trials >= int(self.min_trials_for_full_finetune):
            stages.append(("full", _nonnegative_int(self.full_finetune_steps, "full_finetune_steps"), float(self.full_finetune_learning_rate)))
        for stage, steps, learning_rate in stages:
            self._adapt_stage(x, encoded, stage=stage, steps=steps, learning_rate=learning_rate)
        self.target_calibration_trials_ = int(n_calibration_trials)
        self.model_.eval()
        return self

    def predict_proba(self, features: Sequence | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("The model must be fitted before prediction")
        torch = _torch()
        x = _as_float_matrix(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("features do not match the fitted input dimension")
        batch_size = _positive_int(self.batch_size, "batch_size")
        rows: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device_)
                logits = self.model_(batch, use_adapter=True)
                rows.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(float, copy=False)

    def predict(self, features: Sequence | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "progressive_window_low_rank",
            "hidden_units": int(self.hidden_units),
            "num_layers": int(self.num_layers),
            "adapter_rank": int(self.adapter_rank),
            "source_epochs": int(self.source_epochs),
            "adaptation_history": copy.deepcopy(getattr(self, "adaptation_history_", [])),
            "target_calibration_trials": int(getattr(self, "target_calibration_trials_", 0)),
            "device": str(getattr(self, "device_", self.device)),
        }


__all__ = ["TorchProgressiveWindowClassifier"]
