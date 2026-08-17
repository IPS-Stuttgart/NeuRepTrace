"""Trial-context refinement for the Katja sliding-window endpoint."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from neureptrace.decoding._progressive_sequence_core import _torch
from neureptrace.decoding.katja_window_structure import (
    combine_hierarchical_probabilities,
    conditional_finger_targets,
)


def _TrialContextModule(
    *,
    input_dim: int,
    hidden_units: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    max_windows: int,
):
    torch = _torch()

    class Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(input_dim, hidden_units)
            self.position_embedding = torch.nn.Embedding(max_windows, hidden_units)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
            self.final_norm = torch.nn.LayerNorm(hidden_units)
            self.press_head = torch.nn.Linear(hidden_units, 2)
            self.conditional_finger_head = torch.nn.Linear(hidden_units, 5)
            self.order_head = torch.nn.Linear(hidden_units, 6)
            self.overlap_head = torch.nn.Linear(hidden_units, 1)
            self.template_head = torch.nn.Linear(hidden_units, 2)

        def forward(self, values, padding_mask, *, causal: bool):
            positions = torch.arange(values.shape[1], device=values.device)
            hidden = self.input_projection(values) + self.position_embedding(positions)[None, :, :]
            causal_mask = None
            if causal:
                causal_mask = torch.triu(
                    torch.ones(
                        (values.shape[1], values.shape[1]),
                        dtype=torch.bool,
                        device=values.device,
                    ),
                    diagonal=1,
                )
            hidden = self.encoder(
                hidden,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=bool(causal),
            )
            hidden = self.final_norm(hidden)
            overlap_logit = self.overlap_head(hidden).squeeze(-1)
            return {
                "embedding": hidden,
                "press": self.press_head(hidden),
                "conditional_finger": self.conditional_finger_head(hidden),
                "order": self.order_head(hidden),
                "overlap_logit": overlap_logit,
                "overlap": torch.sigmoid(overlap_logit),
                "template": self.template_head(hidden),
            }

    return Module()


class TorchKatjaTrialContextRefiner:
    """Two-layer trial Transformer with leakage-safe target calibration."""

    def __init__(
        self,
        *,
        hidden_units: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.15,
        max_windows: int = 256,
        source_epochs: int = 6,
        source_refit_all: bool = True,
        adaptation_steps: int = 80,
        source_learning_rate: float = 5e-4,
        adaptation_learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        batch_trials: int = 8,
        order_loss_weight: float = 0.3,
        overlap_loss_weight: float = 0.3,
        template_loss_weight: float = 0.2,
        source_replay_weight: float = 0.1,
        class_balanced_loss: bool = True,
        causal: bool = False,
        random_state: int = 13,
        device: str = "auto",
    ) -> None:
        self.hidden_units = int(hidden_units)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.max_windows = int(max_windows)
        self.source_epochs = int(source_epochs)
        self.source_refit_all = bool(source_refit_all)
        self.adaptation_steps = int(adaptation_steps)
        self.source_learning_rate = float(source_learning_rate)
        self.adaptation_learning_rate = float(adaptation_learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_trials = int(batch_trials)
        self.order_loss_weight = float(order_loss_weight)
        self.overlap_loss_weight = float(overlap_loss_weight)
        self.template_loss_weight = float(template_loss_weight)
        self.source_replay_weight = float(source_replay_weight)
        self.class_balanced_loss = bool(class_balanced_loss)
        self.causal = bool(causal)
        self.random_state = int(random_state)
        self.device = str(device)

    def _resolved_device(self):
        torch = _torch()
        if self.device.lower().strip() in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _seed(self) -> None:
        torch = _torch()
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build(self) -> None:
        self.model_ = _TrialContextModule(
            input_dim=self.input_dim_,
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_windows=self.max_windows,
        ).to(self.device_)

    def _trial_groups(self, indices: np.ndarray) -> list[np.ndarray]:
        keys = np.asarray(
            [f"{self.domain_ids_[index]}::{self.trial_ids_[index]}" for index in indices],
            dtype=object,
        )
        groups: list[np.ndarray] = []
        seen: set[str] = set()
        for key in keys.tolist():
            if key in seen:
                continue
            seen.add(key)
            rows = indices[keys == key]
            if rows.size > self.max_windows:
                raise ValueError(
                    f"Trial {key!r} contains {rows.size} windows, exceeding max_windows={self.max_windows}"
                )
            groups.append(np.sort(rows))
        return groups

    def _padded_batch(self, groups: list[np.ndarray], *, include_labels: bool = True):
        torch = _torch()
        length = max(group.size for group in groups)
        batch = len(groups)
        x = np.zeros((batch, length, self.input_dim_), dtype=np.float32)
        finger = np.zeros((batch, length), dtype=np.int64)
        conditional = np.zeros((batch, length, 5), dtype=np.float32)
        active = np.zeros((batch, length), dtype=bool)
        order = np.zeros((batch, length), dtype=np.int64)
        overlap = np.zeros((batch, length), dtype=np.float32)
        template = np.zeros((batch, length), dtype=np.int64)
        padding = np.ones((batch, length), dtype=bool)
        row_matrix = np.full((batch, length), -1, dtype=np.int64)
        for batch_row, rows in enumerate(groups):
            size = rows.size
            x[batch_row, :size] = self.embeddings_[rows]
            if include_labels:
                forbidden = np.setdiff1d(rows, self.allowed_label_indices_, assume_unique=False)
                if forbidden.size:
                    raise RuntimeError(
                        "Trial-context fitting attempted to access non-calibration target labels"
                    )
                finger[batch_row, :size] = self.finger_labels_[rows]
                conditional[batch_row, :size] = self.conditional_targets_[rows]
                active[batch_row, :size] = self.conditional_active_[rows]
                order[batch_row, :size] = self.order_labels_[rows]
                overlap[batch_row, :size] = self.overlap_targets_[rows]
                template[batch_row, :size] = self.template_labels_[rows]
            padding[batch_row, :size] = False
            row_matrix[batch_row, :size] = rows
        device = self.device_
        return {
            "x": torch.as_tensor(x, dtype=torch.float32, device=device),
            "finger": torch.as_tensor(finger, dtype=torch.long, device=device),
            "conditional": torch.as_tensor(conditional, dtype=torch.float32, device=device),
            "active": torch.as_tensor(active, dtype=torch.bool, device=device),
            "order": torch.as_tensor(order, dtype=torch.long, device=device),
            "overlap": torch.as_tensor(overlap, dtype=torch.float32, device=device),
            "template": torch.as_tensor(template, dtype=torch.long, device=device),
            "padding": torch.as_tensor(padding, dtype=torch.bool, device=device),
            "rows": row_matrix,
        }

    def _loss(self, outputs, batch):
        torch = _torch()
        valid = ~batch["padding"]
        press = (batch["finger"] > 0).long()

        def class_weights(labels, n_classes: int):
            if not self.class_balanced_loss:
                return None
            counts = torch.bincount(labels, minlength=n_classes).to(dtype=torch.float32)
            observed = counts > 0
            weights = torch.zeros_like(counts)
            weights[observed] = labels.numel() / (observed.sum() * counts[observed])
            return weights

        press_labels = press[valid]
        loss = torch.nn.functional.cross_entropy(
            outputs["press"][valid],
            press_labels,
            weight=class_weights(press_labels, 2),
        )
        conditional_mask = valid & batch["active"]
        if torch.any(conditional_mask):
            log_probs = torch.nn.functional.log_softmax(
                outputs["conditional_finger"][conditional_mask], dim=1
            )
            loss = loss - (batch["conditional"][conditional_mask] * log_probs).sum(dim=1).mean()
        order_labels = batch["order"][valid]
        loss = loss + self.order_loss_weight * torch.nn.functional.cross_entropy(
            outputs["order"][valid],
            order_labels,
            weight=class_weights(order_labels, 6),
        )
        loss = loss + self.overlap_loss_weight * torch.nn.functional.smooth_l1_loss(
            outputs["overlap"][valid], batch["overlap"][valid]
        )
        template_labels = batch["template"][valid]
        loss = loss + self.template_loss_weight * torch.nn.functional.cross_entropy(
            outputs["template"][valid],
            template_labels,
            weight=class_weights(template_labels, 2),
        )
        return loss

    def _run_groups(self, groups: list[np.ndarray], optimizer, rng: np.random.Generator) -> float:
        torch = _torch()
        order = rng.permutation(len(groups))
        losses: list[float] = []
        self.model_.train()
        for start in range(0, len(groups), self.batch_trials):
            selected = [groups[index] for index in order[start : start + self.batch_trials]]
            batch = self._padded_batch(selected)
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model_(batch["x"], batch["padding"], causal=self.causal)
            loss = self._loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def fit_source(
        self,
        embeddings: np.ndarray,
        *,
        source_indices: Sequence[int] | np.ndarray,
        domain_ids: np.ndarray,
        trial_ids: np.ndarray,
        finger_labels: np.ndarray,
        press_ratios: np.ndarray,
        order_labels: np.ndarray,
        overlap_targets: np.ndarray,
        template_labels: np.ndarray,
        validation_finger_labels: np.ndarray | None = None,
    ) -> "TorchKatjaTrialContextRefiner":
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embeddings must have shape [windows, features]")
        n_rows = matrix.shape[0]
        vectors = [domain_ids, trial_ids, finger_labels, order_labels, overlap_targets, template_labels]
        if any(np.asarray(values).reshape(-1).size != n_rows for values in vectors):
            raise ValueError("Every context label must align with embeddings")
        self.embeddings_ = matrix
        self.input_dim_ = int(matrix.shape[1])
        self.domain_ids_ = np.asarray(domain_ids).reshape(-1)
        self.trial_ids_ = np.asarray(trial_ids).reshape(-1)
        self.finger_labels_ = np.asarray(finger_labels, dtype=np.int64).reshape(-1)
        self.validation_finger_labels_ = np.asarray(
            self.finger_labels_ if validation_finger_labels is None else validation_finger_labels,
            dtype=np.int64,
        ).reshape(-1)
        if self.validation_finger_labels_.size != n_rows:
            raise ValueError("validation_finger_labels must align with embeddings")
        self.order_labels_ = np.asarray(order_labels, dtype=np.int64).reshape(-1)
        self.overlap_targets_ = np.asarray(overlap_targets, dtype=np.float32).reshape(-1)
        self.template_labels_ = np.asarray(template_labels, dtype=np.int64).reshape(-1)
        self.conditional_targets_, self.conditional_active_ = conditional_finger_targets(
            press_ratios, self.finger_labels_
        )
        self.source_indices_ = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        self.allowed_label_indices_ = np.unique(self.source_indices_)
        domains = np.unique(self.domain_ids_[self.source_indices_])
        if domains.size < 2:
            raise ValueError("Source context fitting requires at least two source subjects")
        validation_domain = domains[self.random_state % domains.size]
        train_indices = self.source_indices_[self.domain_ids_[self.source_indices_] != validation_domain]
        validation_indices = self.source_indices_[self.domain_ids_[self.source_indices_] == validation_domain]
        train_groups = self._trial_groups(train_indices)
        self.device_ = self._resolved_device()
        self._seed()
        self._build()
        torch = _torch()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.source_learning_rate, weight_decay=self.weight_decay
        )
        rng = np.random.default_rng(self.random_state)
        best_state = None
        best_accuracy = -np.inf
        best_epoch = 1
        for epoch in range(self.source_epochs):
            loss = self._run_groups(train_groups, optimizer, rng)
            outputs = self.predict_outputs_indices(validation_indices, _allow_source=True)
            predicted = np.argmax(outputs["probabilities"], axis=1)
            accuracy = float(
                np.mean(predicted == self.validation_finger_labels_[validation_indices])
            )
            print(
                f"trial context source epoch {epoch + 1}/{self.source_epochs}: "
                f"loss={loss:.5f} finger_accuracy={accuracy:.5f}",
                flush=True,
            )
            if accuracy > best_accuracy + 1e-5:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model_.state_dict().items()
                }
        self.best_source_epoch_ = int(best_epoch)
        if self.source_refit_all:
            self._seed()
            self._build()
            optimizer = torch.optim.AdamW(
                self.model_.parameters(),
                lr=self.source_learning_rate,
                weight_decay=self.weight_decay,
            )
            refit_rng = np.random.default_rng(self.random_state)
            source_groups = self._trial_groups(self.source_indices_)
            for epoch in range(best_epoch):
                loss = self._run_groups(source_groups, optimizer, refit_rng)
                print(
                    f"trial context source refit epoch {epoch + 1}/{best_epoch}: "
                    f"loss={loss:.5f}",
                    flush=True,
                )
        elif best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_source_validation_accuracy_ = float(best_accuracy)
        self.source_state_ = {
            name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()
        }
        self.model_.eval()
        return self

    def clone_source(self, *, random_state: int | None = None) -> "TorchKatjaTrialContextRefiner":
        if not hasattr(self, "source_state_"):
            raise RuntimeError("fit_source must run before clone_source")
        clone = copy.copy(self)
        clone.random_state = self.random_state if random_state is None else int(random_state)
        clone._seed()
        clone._build()
        clone.model_.load_state_dict(self.source_state_)
        return clone

    def register_target_calibration_labels(
        self,
        calibration_indices: Sequence[int] | np.ndarray,
        *,
        finger_labels: np.ndarray,
        press_ratios: np.ndarray,
        order_labels: np.ndarray,
        overlap_targets: np.ndarray,
        template_labels: np.ndarray,
    ) -> "TorchKatjaTrialContextRefiner":
        indices = np.asarray(calibration_indices, dtype=np.int64).reshape(-1)
        arrays = [
            np.asarray(finger_labels).reshape(-1),
            np.asarray(order_labels).reshape(-1),
            np.asarray(overlap_targets).reshape(-1),
            np.asarray(template_labels).reshape(-1),
        ]
        if indices.size == 0 or any(array.size != indices.size for array in arrays):
            raise ValueError("Calibration labels must align with nonempty calibration_indices")
        conditional, active = conditional_finger_targets(press_ratios, arrays[0])
        self.finger_labels_ = self.finger_labels_.copy()
        self.order_labels_ = self.order_labels_.copy()
        self.overlap_targets_ = self.overlap_targets_.copy()
        self.template_labels_ = self.template_labels_.copy()
        self.conditional_targets_ = self.conditional_targets_.copy()
        self.conditional_active_ = self.conditional_active_.copy()
        self.finger_labels_[indices] = np.asarray(arrays[0], dtype=np.int64)
        self.order_labels_[indices] = np.asarray(arrays[1], dtype=np.int64)
        self.overlap_targets_[indices] = np.asarray(arrays[2], dtype=np.float32)
        self.template_labels_[indices] = np.asarray(arrays[3], dtype=np.int64)
        self.conditional_targets_[indices] = conditional
        self.conditional_active_[indices] = active
        self.registered_calibration_indices_ = indices.copy()
        return self

    def adapt_target_indices(
        self,
        calibration_indices: Sequence[int] | np.ndarray,
    ) -> "TorchKatjaTrialContextRefiner":
        torch = _torch()
        calibration = np.asarray(calibration_indices, dtype=np.int64).reshape(-1)
        if calibration.size == 0:
            raise ValueError("calibration_indices must not be empty")
        registered = getattr(self, "registered_calibration_indices_", None)
        if registered is None or not np.array_equal(np.sort(calibration), np.sort(registered)):
            raise RuntimeError("Target context adaptation requires registered calibration labels")
        self.allowed_label_indices_ = np.unique(np.concatenate((self.source_indices_, calibration)))
        groups = self._trial_groups(calibration)
        source_groups = self._trial_groups(self.source_indices_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.adaptation_learning_rate, weight_decay=self.weight_decay
        )
        rng = np.random.default_rng(self.random_state + 104729)
        self.model_.train()
        for _step in range(self.adaptation_steps):
            target_group = [groups[int(rng.integers(len(groups)))]]
            batch = self._padded_batch(target_group)
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model_(batch["x"], batch["padding"], causal=self.causal)
            loss = self._loss(outputs, batch)
            if self.source_replay_weight > 0.0:
                source_group = [source_groups[int(rng.integers(len(source_groups)))]]
                source_batch = self._padded_batch(source_group)
                source_outputs = self.model_(
                    source_batch["x"], source_batch["padding"], causal=self.causal
                )
                loss = loss + self.source_replay_weight * self._loss(source_outputs, source_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
            optimizer.step()
        self.model_.eval()
        return self

    def predict_outputs_indices(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        _allow_source: bool = False,
    ) -> dict[str, np.ndarray]:
        torch = _torch()
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        if requested.size == 0:
            raise ValueError("indices must not be empty")
        if _allow_source and np.setdiff1d(requested, self.source_indices_).size:
            raise RuntimeError("Internal source validation requested non-source rows")
        groups = self._trial_groups(requested)
        row_to_output = {int(row): position for position, row in enumerate(requested.tolist())}
        result: dict[str, np.ndarray] = {}
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(groups), self.batch_trials):
                selected = groups[start : start + self.batch_trials]
                batch = self._padded_batch(selected, include_labels=False)
                outputs = self.model_(batch["x"], batch["padding"], causal=self.causal)
                for batch_row, rows in enumerate(selected):
                    positions = np.asarray([row_to_output[int(row)] for row in rows], dtype=int)
                    for name, values in outputs.items():
                        shape = (requested.size, *values.shape[2:])
                        result.setdefault(name, np.empty(shape, dtype=np.float32))
                        result[name][positions] = values[batch_row, : rows.size].detach().cpu().numpy()
        press = torch.softmax(torch.as_tensor(result["press"]), dim=1).numpy()
        conditional = torch.softmax(torch.as_tensor(result["conditional_finger"]), dim=1).numpy()
        result["press_probabilities"] = press
        result["conditional_finger_probabilities"] = conditional
        result["probabilities"] = combine_hierarchical_probabilities(press, conditional)
        result["order_probabilities"] = torch.softmax(torch.as_tensor(result["order"]), dim=1).numpy()
        result["template_probabilities"] = torch.softmax(torch.as_tensor(result["template"]), dim=1).numpy()
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "katja_trial_context_transformer",
            "causal": bool(self.causal),
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "source_epochs": self.source_epochs,
            "best_source_epoch": int(getattr(self, "best_source_epoch_", 0)),
            "source_refit_all": bool(self.source_refit_all),
            "adaptation_steps": self.adaptation_steps,
            "best_source_validation_accuracy": float(
                getattr(self, "best_source_validation_accuracy_", np.nan)
            ),
            "class_balanced_loss": bool(self.class_balanced_loss),
        }


__all__ = ["TorchKatjaTrialContextRefiner"]
