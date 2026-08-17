"""Progressive multi-task adaptation for raw temporal MEG windows."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from neureptrace.decoding._progressive_sequence_core import _torch
from neureptrace.decoding.katja_window_structure import (
    balanced_window_sampling_weights,
    combine_hierarchical_probabilities,
    conditional_finger_targets,
)


def _TemporalWindowModule(
    *,
    n_channels: int,
    n_finger_classes: int,
    n_sequence_classes: int,
    n_order_classes: int,
    hidden_units: int,
    num_blocks: int,
    adapter_rank: int,
    adapter_alpha: float,
    dropout: float,
    hierarchical: bool = False,
    n_source_adapters: int = 0,
    adapter_kind: str = "low_rank",
):
    torch = _torch()

    class TemporalBlock(torch.nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            groups = 8 if hidden_units % 8 == 0 else 1
            self.network = torch.nn.Sequential(
                torch.nn.GroupNorm(groups, hidden_units),
                torch.nn.GELU(),
                torch.nn.Conv1d(
                    hidden_units,
                    hidden_units,
                    kernel_size=5,
                    padding=2 * dilation,
                    dilation=dilation,
                ),
                torch.nn.Dropout(dropout),
                torch.nn.GroupNorm(groups, hidden_units),
                torch.nn.GELU(),
                torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
                torch.nn.Dropout(dropout),
            )

        def forward(self, values):
            return values + self.network(values)

    class Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_projection = torch.nn.Linear(n_channels, hidden_units)
            self.adapter_down = torch.nn.Linear(n_channels, adapter_rank, bias=False)
            self.adapter_up = torch.nn.Linear(adapter_rank, hidden_units, bias=False)
            self.adapter_scale = float(adapter_alpha) / float(adapter_rank)
            self.adapter_kind = str(adapter_kind)
            if self.adapter_kind not in {"low_rank", "channel_affine_residual"}:
                raise ValueError("adapter_kind must be low_rank or channel_affine_residual")
            self.source_adapters = torch.nn.ModuleList()
            self.source_channel_scales = torch.nn.ParameterList()
            self.source_channel_biases = torch.nn.ParameterList()
            self.source_residuals = torch.nn.ModuleList()
            if self.adapter_kind == "low_rank":
                self.source_adapters.extend(
                    torch.nn.ModuleDict(
                        {
                            "down": torch.nn.Linear(n_channels, adapter_rank, bias=False),
                            "up": torch.nn.Linear(adapter_rank, hidden_units, bias=False),
                        }
                    )
                    for _ in range(int(n_source_adapters))
                )
            else:
                for _ in range(int(n_source_adapters)):
                    self.source_channel_scales.append(
                        torch.nn.Parameter(torch.ones(n_channels))
                    )
                    self.source_channel_biases.append(
                        torch.nn.Parameter(torch.zeros(n_channels))
                    )
                    self.source_residuals.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(n_channels, hidden_units),
                            torch.nn.GELU(),
                            torch.nn.Linear(hidden_units, hidden_units),
                        )
                    )
            if self.adapter_kind == "channel_affine_residual":
                self.target_channel_scale = torch.nn.Parameter(torch.ones(n_channels))
                self.target_channel_bias = torch.nn.Parameter(torch.zeros(n_channels))
                self.target_residual = torch.nn.Sequential(
                    torch.nn.Linear(n_channels, hidden_units),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_units, hidden_units),
                )
            self.temporal_blocks = torch.nn.ModuleList(TemporalBlock(2**index) for index in range(num_blocks))
            self.pool_projection = torch.nn.Sequential(
                torch.nn.Linear(hidden_units * 2, hidden_units),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.final_norm = torch.nn.LayerNorm(hidden_units)
            self.finger_head = torch.nn.Linear(hidden_units, n_finger_classes)
            self.sequence_head = torch.nn.Linear(hidden_units, n_sequence_classes)
            self.order_head = torch.nn.Linear(hidden_units, n_order_classes)
            self.overlap_head = torch.nn.Linear(hidden_units, 1)
            self.hierarchical = bool(hierarchical)
            if self.hierarchical:
                self.press_head = torch.nn.Linear(hidden_units, 2)
                self.conditional_finger_head = torch.nn.Linear(hidden_units, 5)
            torch.nn.init.zeros_(self.adapter_up.weight)
            if self.adapter_kind == "channel_affine_residual":
                torch.nn.init.zeros_(self.target_residual[-1].weight)
                torch.nn.init.zeros_(self.target_residual[-1].bias)
            for source_adapter in self.source_adapters:
                torch.nn.init.zeros_(source_adapter["up"].weight)
            for source_residual in self.source_residuals:
                torch.nn.init.zeros_(source_residual[-1].weight)
                torch.nn.init.zeros_(source_residual[-1].bias)

        def _source_adapter(self, values, adapter_ids):
            contribution = torch.zeros(
                (*values.shape[:2], hidden_units),
                dtype=values.dtype,
                device=values.device,
            )
            n_adapters = (
                len(self.source_adapters)
                if self.adapter_kind == "low_rank"
                else len(self.source_residuals)
            )
            for adapter_index in range(n_adapters):
                selected = adapter_ids == adapter_index
                if torch.any(selected):
                    if self.adapter_kind == "low_rank":
                        adapter = self.source_adapters[adapter_index]
                        contribution[selected] = self.adapter_scale * adapter["up"](
                            adapter["down"](values[selected])
                        )
                    else:
                        adapted = (
                            values[selected] * self.source_channel_scales[adapter_index]
                            + self.source_channel_biases[adapter_index]
                        )
                        contribution[selected] = (
                            self.base_projection(adapted)
                            - self.base_projection(values[selected])
                            + self.source_residuals[adapter_index](adapted)
                        )
            return contribution

        def forward(self, values, *, use_adapter: bool, source_adapter_ids=None):
            adapted_values = values
            if use_adapter and self.adapter_kind == "channel_affine_residual":
                adapted_values = values * self.target_channel_scale + self.target_channel_bias
            hidden = self.base_projection(adapted_values)
            if source_adapter_ids is not None and (
                len(self.source_adapters) or len(self.source_residuals)
            ):
                hidden = hidden + self._source_adapter(values, source_adapter_ids)
            if use_adapter:
                if self.adapter_kind == "low_rank":
                    hidden = hidden + self.adapter_scale * self.adapter_up(self.adapter_down(values))
                else:
                    hidden = hidden + self.target_residual(adapted_values)
            hidden = torch.nn.functional.gelu(hidden).transpose(1, 2)
            for block in self.temporal_blocks:
                hidden = block(hidden)
            pooled = torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=1)
            pooled = self.final_norm(self.pool_projection(pooled))
            overlap_logit = self.overlap_head(pooled).squeeze(1)
            outputs = {
                "finger": self.finger_head(pooled),
                "sequence": self.sequence_head(pooled),
                "order": self.order_head(pooled),
                "overlap_logit": overlap_logit,
                "overlap": torch.sigmoid(overlap_logit),
                "embedding": pooled,
            }
            if self.hierarchical:
                outputs["press"] = self.press_head(pooled)
                outputs["conditional_finger"] = self.conditional_finger_head(pooled)
            return outputs

    return Module()


class TorchProgressiveTemporalWindowClassifier:
    """Raw-window multi-task model with a pre-encoder low-rank adapter."""

    def __init__(
        self,
        *,
        hidden_units: int = 96,
        num_blocks: int = 3,
        adapter_rank: int = 8,
        adapter_alpha: float = 8.0,
        source_epochs: int = 10,
        source_validation_patience: int = 3,
        source_refit_all: bool = True,
        source_learning_rate: float = 1e-3,
        adapter_steps: int = 100,
        last_block_steps: int = 80,
        full_finetune_steps: int = 80,
        adaptation_learning_rate: float = 2e-3,
        full_finetune_learning_rate: float = 2e-4,
        batch_size: int = 512,
        weight_decay: float = 1e-4,
        dropout: float = 0.15,
        feature_noise_std: float = 0.01,
        source_replay_weight: float = 0.1,
        l2sp_weight: float = 1e-5,
        sequence_loss_weight: float = 0.15,
        order_loss_weight: float = 0.30,
        overlap_loss_weight: float = 0.30,
        press_loss_weight: float = 1.0,
        conditional_finger_loss_weight: float = 1.0,
        finger_auxiliary_loss_weight: float = 0.25,
        min_trials_for_last_block: int = 8,
        min_trials_for_full_finetune: int = 12,
        hierarchical: bool = False,
        balanced_sampling: bool = False,
        subject_specific_normalization: bool = False,
        source_specific_adapters: bool = False,
        source_selection_metric: str = "loss",
        source_validation_domain: Any | None = None,
        adapter_kind: str = "low_rank",
        random_state: int = 13,
        device: str = "auto",
    ) -> None:
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.source_epochs = source_epochs
        self.source_validation_patience = source_validation_patience
        self.source_refit_all = source_refit_all
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
        self.sequence_loss_weight = sequence_loss_weight
        self.order_loss_weight = order_loss_weight
        self.overlap_loss_weight = overlap_loss_weight
        self.press_loss_weight = press_loss_weight
        self.conditional_finger_loss_weight = conditional_finger_loss_weight
        self.finger_auxiliary_loss_weight = finger_auxiliary_loss_weight
        self.min_trials_for_last_block = min_trials_for_last_block
        self.min_trials_for_full_finetune = min_trials_for_full_finetune
        self.hierarchical = hierarchical
        self.balanced_sampling = balanced_sampling
        self.subject_specific_normalization = subject_specific_normalization
        self.source_specific_adapters = source_specific_adapters
        if source_selection_metric not in {"loss", "finger_accuracy"}:
            raise ValueError("source_selection_metric must be loss or finger_accuracy")
        self.source_selection_metric = source_selection_metric
        self.source_validation_domain = source_validation_domain
        if adapter_kind not in {"low_rank", "channel_affine_residual"}:
            raise ValueError("adapter_kind must be low_rank or channel_affine_residual")
        self.adapter_kind = adapter_kind
        self.random_state = random_state
        self.device = device

    def _resolved_device(self):
        torch = _torch()
        requested = str(self.device).lower().strip()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _seed_torch(self) -> None:
        torch = _torch()
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

    def _build_model(self) -> None:
        self.model_ = _TemporalWindowModule(
            n_channels=self.n_channels_,
            n_finger_classes=self.finger_classes_.size,
            n_sequence_classes=self.sequence_classes_.size,
            n_order_classes=self.order_classes_.size,
            hidden_units=int(self.hidden_units),
            num_blocks=int(self.num_blocks),
            adapter_rank=int(self.adapter_rank),
            adapter_alpha=float(self.adapter_alpha),
            dropout=float(self.dropout),
            hierarchical=bool(self.hierarchical),
            n_source_adapters=len(getattr(self, "source_domain_values_", ())) if self.source_specific_adapters else 0,
            adapter_kind=str(self.adapter_kind),
        ).to(self.device_)

    @staticmethod
    def _encode(values: np.ndarray, classes: np.ndarray, *, name: str) -> np.ndarray:
        mapping = {value: index for index, value in enumerate(classes.tolist())}
        try:
            return np.asarray([mapping[value] for value in values.tolist()], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"{name} contains class {error.args[0]!r}, absent from source data") from error

    def _normalized_batch(self, indices: np.ndarray):
        values = np.asarray(self.window_store_[indices], dtype=np.float32)
        if bool(self.subject_specific_normalization) and hasattr(self, "row_domains_"):
            means = np.empty((indices.size, self.n_channels_), dtype=np.float32)
            stds = np.empty_like(means)
            for row, index in enumerate(indices.tolist()):
                domain = self.row_domains_[index]
                if hasattr(self, "target_normalization_domain_") and domain == self.target_normalization_domain_:
                    means[row] = self.target_sensor_mean_
                    stds[row] = self.target_sensor_std_
                else:
                    position = self.domain_to_normalization_row_.get(domain)
                    if position is None:
                        means[row] = self.sensor_mean_
                        stds[row] = self.sensor_std_
                    else:
                        means[row] = self.subject_sensor_means_[position]
                        stds[row] = self.subject_sensor_stds_[position]
            return (values - means[:, None, :]) / stds[:, None, :]
        return (values - self.sensor_mean_[None, None, :]) / self.sensor_std_[None, None, :]

    def _labels_for(self, indices: np.ndarray) -> tuple[np.ndarray, ...]:
        if hasattr(self, "allowed_label_indices_"):
            forbidden = np.setdiff1d(indices, self.allowed_label_indices_, assume_unique=False)
            if forbidden.size:
                raise RuntimeError(
                    "Attempted to access target labels outside source/calibration rows; "
                    f"first forbidden row is {int(forbidden[0])}"
                )
        return (
            self.finger_encoded_[indices],
            self.sequence_encoded_[indices],
            self.order_encoded_[indices],
            self.overlap_targets_[indices],
            self.conditional_finger_targets_[indices],
            self.conditional_finger_active_[indices],
        )

    def _loss(self, outputs, finger, sequence, order, overlap, conditional_finger, conditional_active):
        torch = _torch()
        if bool(self.hierarchical):
            press = (finger > 0).long()
            loss = float(self.press_loss_weight) * torch.nn.functional.cross_entropy(outputs["press"], press)
            if torch.any(conditional_active):
                log_conditional = torch.nn.functional.log_softmax(outputs["conditional_finger"][conditional_active], dim=1)
                conditional_loss = -(conditional_finger[conditional_active] * log_conditional).sum(dim=1).mean()
                loss = loss + float(self.conditional_finger_loss_weight) * conditional_loss
            loss = loss + float(self.finger_auxiliary_loss_weight) * torch.nn.functional.cross_entropy(outputs["finger"], finger)
        else:
            loss = torch.nn.functional.cross_entropy(outputs["finger"], finger)
        loss = loss + float(self.sequence_loss_weight) * torch.nn.functional.cross_entropy(outputs["sequence"], sequence)
        loss = loss + float(self.order_loss_weight) * torch.nn.functional.cross_entropy(outputs["order"], order)
        loss = loss + float(self.overlap_loss_weight) * torch.nn.functional.smooth_l1_loss(outputs["overlap"], overlap)
        return loss

    def _tensor_batch(self, indices: np.ndarray):
        torch = _torch()
        x = torch.as_tensor(self._normalized_batch(indices), dtype=torch.float32, device=self.device_)
        finger, sequence, order, overlap, conditional_finger, conditional_active = self._labels_for(indices)
        return (
            x,
            torch.as_tensor(finger, dtype=torch.long, device=self.device_),
            torch.as_tensor(sequence, dtype=torch.long, device=self.device_),
            torch.as_tensor(order, dtype=torch.long, device=self.device_),
            torch.as_tensor(overlap, dtype=torch.float32, device=self.device_),
            torch.as_tensor(conditional_finger, dtype=torch.float32, device=self.device_),
            torch.as_tensor(conditional_active, dtype=torch.bool, device=self.device_),
        )

    def _source_adapter_ids(self, indices: np.ndarray):
        if not bool(self.source_specific_adapters):
            return None
        torch = _torch()
        ids = np.asarray([self.source_domain_to_adapter_.get(self.row_domains_[index], -1) for index in indices], dtype=np.int64)
        return torch.as_tensor(ids, dtype=torch.long, device=self.device_)

    def _augment(self, values):
        torch = _torch()
        if float(self.feature_noise_std) <= 0.0:
            return values
        return values + float(self.feature_noise_std) * torch.randn_like(values)

    def _source_optimizer(self):
        torch = _torch()
        return torch.optim.AdamW(
            (parameter for parameter in self.model_.parameters() if parameter.requires_grad),
            lr=float(self.source_learning_rate),
            weight_decay=float(self.weight_decay),
        )

    def _set_source_trainable(self) -> None:
        for name, parameter in self.model_.named_parameters():
            is_target_adapter = name.startswith("adapter_") or name.startswith("target_")
            parameter.requires_grad_(not is_target_adapter)

    def _initialize_target_adapter_from_sources(self) -> None:
        torch = _torch()
        if not bool(self.source_specific_adapters):
            return
        if str(self.adapter_kind) == "low_rank" and not len(self.model_.source_adapters):
            return
        with torch.no_grad():
            if str(self.adapter_kind) == "low_rank":
                effective = torch.stack(
                    [
                        self.model_.adapter_scale
                        * adapter["up"].weight
                        @ adapter["down"].weight
                        for adapter in self.model_.source_adapters
                    ]
                ).mean(dim=0)
                left, singular_values, right = torch.linalg.svd(
                    effective, full_matrices=False
                )
                rank = min(
                    self.model_.adapter_down.weight.shape[0], singular_values.numel()
                )
                factor = torch.sqrt(
                    singular_values[:rank] / float(self.model_.adapter_scale)
                )
                down = torch.zeros_like(self.model_.adapter_down.weight)
                up = torch.zeros_like(self.model_.adapter_up.weight)
                down[:rank] = factor[:, None] * right[:rank]
                up[:, :rank] = left[:, :rank] * factor[None, :]
                self.model_.adapter_down.weight.copy_(down)
                self.model_.adapter_up.weight.copy_(up)
            else:
                if not len(self.model_.source_residuals):
                    return
                self.model_.target_channel_scale.copy_(
                    torch.stack(list(self.model_.source_channel_scales)).mean(dim=0)
                )
                self.model_.target_channel_bias.copy_(
                    torch.stack(list(self.model_.source_channel_biases)).mean(dim=0)
                )
                for parameter_index, target_parameter in enumerate(
                    self.model_.target_residual.parameters()
                ):
                    target_parameter.copy_(
                        torch.stack(
                            [
                                list(source.parameters())[parameter_index]
                                for source in self.model_.source_residuals
                            ]
                        ).mean(dim=0)
                    )

    def _run_source_epoch(self, indices: np.ndarray, optimizer, rng: np.random.Generator) -> float:
        torch = _torch()
        if bool(self.balanced_sampling):
            probabilities = self.source_sampling_weights_[indices]
            probabilities = probabilities / probabilities.sum()
            order_indices = rng.choice(indices, size=indices.size, replace=True, p=probabilities)
        else:
            order_indices = rng.permutation(indices)
        running_loss = 0.0
        n_batches = 0
        self.model_.train()
        for start in range(0, order_indices.size, int(self.batch_size)):
            batch_indices = order_indices[start : start + int(self.batch_size)]
            x, finger, sequence, order, overlap, conditional_finger, conditional_active = self._tensor_batch(batch_indices)
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model_(
                self._augment(x),
                use_adapter=False,
                source_adapter_ids=self._source_adapter_ids(batch_indices),
            )
            loss = self._loss(
                outputs,
                finger,
                sequence,
                order,
                overlap,
                conditional_finger,
                conditional_active,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            n_batches += 1
        return running_loss / max(1, n_batches)

    def _source_validation_metrics(self, indices: np.ndarray) -> tuple[float, float]:
        torch = _torch()
        losses: list[float] = []
        correct = 0
        count = 0
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, indices.size, int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x, finger, sequence, order, overlap, conditional_finger, conditional_active = self._tensor_batch(batch_indices)
                outputs = self.model_(
                    x,
                    use_adapter=False,
                    source_adapter_ids=self._source_adapter_ids(batch_indices),
                )
                losses.append(
                    float(
                        self._loss(
                            outputs,
                            finger,
                            sequence,
                            order,
                            overlap,
                            conditional_finger,
                            conditional_active,
                        ).detach().cpu()
                    )
                )
                if bool(self.hierarchical):
                    press = torch.softmax(outputs["press"], dim=1).detach().cpu().numpy()
                    conditional = torch.softmax(outputs["conditional_finger"], dim=1).detach().cpu().numpy()
                    probabilities = combine_hierarchical_probabilities(press, conditional)
                    predicted = np.argmax(probabilities, axis=1)
                else:
                    predicted = torch.argmax(outputs["finger"], dim=1).detach().cpu().numpy()
                correct += int(
                    np.sum(predicted == self.validation_finger_encoded_[batch_indices])
                )
                count += int(finger.shape[0])
        return float(np.mean(losses)), float(correct / max(1, count))

    def _source_validation_loss(self, indices: np.ndarray) -> float:
        return self._source_validation_metrics(indices)[0]

    def fit_source(
        self,
        window_store: np.ndarray,
        *,
        source_indices: Sequence[int] | np.ndarray,
        source_domains: np.ndarray | None = None,
        finger_labels: np.ndarray,
        sequence_labels: np.ndarray,
        order_labels: np.ndarray,
        overlap_targets: np.ndarray,
        sensor_mean: np.ndarray,
        sensor_std: np.ndarray,
        press_ratios: np.ndarray | None = None,
        trial_ids: np.ndarray | None = None,
        subject_sensor_domains: np.ndarray | None = None,
        subject_sensor_means: np.ndarray | None = None,
        subject_sensor_stds: np.ndarray | None = None,
        validation_finger_labels: np.ndarray | None = None,
    ) -> "TorchProgressiveTemporalWindowClassifier":
        if window_store.ndim != 3 or window_store.shape[0] == 0:
            raise ValueError("window_store must have shape [windows, time, channels]")
        n_rows = window_store.shape[0]
        vectors = [np.asarray(values).reshape(-1) for values in (finger_labels, sequence_labels, order_labels, overlap_targets)]
        if any(values.shape[0] != n_rows for values in vectors):
            raise ValueError("Every label array must align with window_store rows")
        source_indices = np.asarray(source_indices, dtype=int).reshape(-1)
        if source_indices.size == 0 or np.any(source_indices < 0) or np.any(source_indices >= n_rows):
            raise ValueError("source_indices are empty or out of range")
        self.window_store_ = window_store
        self.source_indices_ = source_indices
        self.finger_classes_ = np.unique(vectors[0][source_indices])
        self.sequence_classes_ = np.unique(vectors[1][source_indices])
        self.order_classes_ = np.unique(vectors[2][source_indices])
        self.finger_encoded_ = self._encode(vectors[0], self.finger_classes_, name="finger_labels")
        validation_labels = (
            vectors[0]
            if validation_finger_labels is None
            else np.asarray(validation_finger_labels).reshape(-1)
        )
        if validation_labels.shape[0] != n_rows:
            raise ValueError("validation_finger_labels must align with window_store rows")
        self.validation_finger_encoded_ = self._encode(
            validation_labels,
            self.finger_classes_,
            name="validation_finger_labels",
        )
        self.sequence_encoded_ = self._encode(vectors[1], self.sequence_classes_, name="sequence_labels")
        self.order_encoded_ = self._encode(vectors[2], self.order_classes_, name="order_labels")
        self.overlap_targets_ = np.asarray(vectors[3], dtype=np.float32)
        if press_ratios is None:
            synthetic_ratios = np.zeros((n_rows, 6), dtype=np.float32)
            synthetic_ratios[np.arange(n_rows), np.asarray(vectors[0], dtype=np.int64)] = 1.0
            press_ratios = synthetic_ratios
        self.conditional_finger_targets_, self.conditional_finger_active_ = conditional_finger_targets(
            press_ratios,
            np.asarray(vectors[0], dtype=np.int64),
        )
        self.trial_ids_ = np.arange(n_rows, dtype=np.int64) if trial_ids is None else np.asarray(trial_ids).reshape(-1)
        if self.trial_ids_.shape[0] != n_rows:
            raise ValueError("trial_ids must align with window_store rows")
        self.sensor_mean_ = np.asarray(sensor_mean, dtype=np.float32).reshape(-1)
        self.sensor_std_ = np.asarray(sensor_std, dtype=np.float32).reshape(-1)
        self.sensor_std_[self.sensor_std_ < 1e-6] = 1.0
        self.n_channels_ = int(window_store.shape[2])
        if self.sensor_mean_.size != self.n_channels_ or self.sensor_std_.size != self.n_channels_:
            raise ValueError("sensor_mean/sensor_std must contain one value per channel")
        if source_domains is None:
            self.row_domains_ = np.zeros(n_rows, dtype=np.int64)
        else:
            self.row_domains_ = np.asarray(source_domains).reshape(-1)
            if self.row_domains_.shape[0] != n_rows:
                raise ValueError("source_domains must align with window_store rows")
        self.source_domain_values_ = np.unique(self.row_domains_[source_indices])
        self.source_domain_to_adapter_ = {
            value: position for position, value in enumerate(self.source_domain_values_.tolist())
        }
        if bool(self.subject_specific_normalization):
            if subject_sensor_domains is None or subject_sensor_means is None or subject_sensor_stds is None:
                raise ValueError(
                    "subject-specific normalization requires subject sensor domains, means, and standard deviations"
                )
            self.normalization_domains_ = np.asarray(subject_sensor_domains).reshape(-1)
            self.subject_sensor_means_ = np.asarray(subject_sensor_means, dtype=np.float32)
            self.subject_sensor_stds_ = np.asarray(subject_sensor_stds, dtype=np.float32)
            expected = (self.normalization_domains_.size, self.n_channels_)
            if self.subject_sensor_means_.shape != expected or self.subject_sensor_stds_.shape != expected:
                raise ValueError(f"subject sensor moments must have shape {expected}")
            self.subject_sensor_stds_[self.subject_sensor_stds_ < 1e-6] = 1.0
            self.domain_to_normalization_row_ = {
                value: position for position, value in enumerate(self.normalization_domains_.tolist())
            }
        self.allowed_label_indices_ = np.unique(source_indices)
        self.source_sampling_weights_ = np.zeros(n_rows, dtype=np.float64)
        if bool(self.balanced_sampling):
            self.source_sampling_weights_[source_indices] = balanced_window_sampling_weights(
                self.row_domains_[source_indices],
                self.trial_ids_[source_indices],
                np.asarray(vectors[0])[source_indices],
            )
        else:
            self.source_sampling_weights_[source_indices] = 1.0 / source_indices.size
        self.device_ = self._resolved_device()
        self._seed_torch()
        self._build_model()
        self._set_source_trainable()
        rng = np.random.default_rng(int(self.random_state))
        if source_domains is None:
            shuffled = rng.permutation(source_indices)
            n_validation = min(source_indices.size - 1, max(1, int(round(0.1 * source_indices.size))))
            validation_indices = shuffled[:n_validation]
            training_indices = shuffled[n_validation:]
            self.source_validation_mode_ = "source_window_fallback"
            self.source_validation_domain_ = None
        else:
            domains = self.row_domains_
            unique_domains = np.unique(domains[source_indices])
            if unique_domains.size < 2:
                raise ValueError("Source held-subject validation requires at least two source domains")
            if self.source_validation_domain is None:
                validation_domain = unique_domains[int(self.random_state) % unique_domains.size]
            else:
                matches = unique_domains[unique_domains == self.source_validation_domain]
                if matches.size != 1:
                    raise ValueError(
                        f"source_validation_domain={self.source_validation_domain!r} is not in source domains"
                    )
                validation_domain = matches[0]
            validation_indices = source_indices[domains[source_indices] == validation_domain]
            training_indices = source_indices[domains[source_indices] != validation_domain]
            self.source_validation_mode_ = "heldout_source_subject"
            self.source_validation_domain_ = validation_domain.item() if hasattr(validation_domain, "item") else validation_domain
        optimizer = self._source_optimizer()
        best_state = None
        best_validation_loss = np.inf
        best_validation_accuracy = -np.inf
        best_selection_value = np.inf if self.source_selection_metric == "loss" else -np.inf
        best_epoch = 1
        patience_left = int(self.source_validation_patience)
        for epoch in range(int(self.source_epochs)):
            training_loss = self._run_source_epoch(training_indices, optimizer, rng)
            validation_loss, validation_accuracy = self._source_validation_metrics(validation_indices)
            print(
                f"temporal source select epoch {epoch + 1}/{self.source_epochs}: "
                f"train={training_loss:.5f} validation={validation_loss:.5f} "
                f"finger_accuracy={validation_accuracy:.5f}",
                flush=True,
            )
            selection_value = validation_loss if self.source_selection_metric == "loss" else validation_accuracy
            improved = (
                selection_value + 1e-5 < best_selection_value
                if self.source_selection_metric == "loss"
                else selection_value > best_selection_value + 1e-5
            )
            if improved:
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                best_selection_value = selection_value
                best_epoch = epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
                patience_left = int(self.source_validation_patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        self.best_source_epoch_ = int(best_epoch)
        self.best_source_validation_loss_ = float(best_validation_loss)
        if bool(self.source_refit_all):
            self._seed_torch()
            self._build_model()
            self._set_source_trainable()
            optimizer = self._source_optimizer()
            refit_rng = np.random.default_rng(int(self.random_state))
            for epoch in range(best_epoch):
                refit_loss = self._run_source_epoch(source_indices, optimizer, refit_rng)
                print(
                    f"temporal source refit epoch {epoch + 1}/{best_epoch}: loss={refit_loss:.5f}",
                    flush=True,
                )
        elif best_state is not None:
            self.model_.load_state_dict(best_state)
        self._initialize_target_adapter_from_sources()
        self.source_state_ = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
        self.best_source_validation_accuracy_ = float(best_validation_accuracy)
        self.model_.eval()
        return self

    def clone_source(self, *, random_state: int | None = None) -> "TorchProgressiveTemporalWindowClassifier":
        if not hasattr(self, "source_state_"):
            raise RuntimeError("fit_source must be called before clone_source")
        clone = copy.copy(self)
        if random_state is not None:
            clone.random_state = int(random_state)
        clone._seed_torch()
        clone._build_model()
        clone.model_.load_state_dict(self.source_state_)
        clone.adaptation_history_ = []
        clone.model_.eval()
        return clone

    def register_target_calibration_labels(
        self,
        calibration_indices: Sequence[int] | np.ndarray,
        *,
        finger_labels: np.ndarray,
        sequence_labels: np.ndarray,
        order_labels: np.ndarray,
        overlap_targets: np.ndarray,
        press_ratios: np.ndarray,
    ) -> "TorchProgressiveTemporalWindowClassifier":
        """Register labels for explicit calibration rows, never evaluation rows."""

        indices = np.asarray(calibration_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.unique(indices).size != indices.size:
            raise ValueError("calibration_indices must contain unique rows")
        arrays = [
            np.asarray(finger_labels).reshape(-1),
            np.asarray(sequence_labels).reshape(-1),
            np.asarray(order_labels).reshape(-1),
            np.asarray(overlap_targets).reshape(-1),
        ]
        if any(array.size != indices.size for array in arrays):
            raise ValueError("Every calibration label array must align with calibration_indices")
        encoded_finger = self._encode(arrays[0], self.finger_classes_, name="target finger_labels")
        encoded_sequence = self._encode(arrays[1], self.sequence_classes_, name="target sequence_labels")
        encoded_order = self._encode(arrays[2], self.order_classes_, name="target order_labels")
        conditional, active = conditional_finger_targets(press_ratios, arrays[0])
        self.finger_encoded_ = self.finger_encoded_.copy()
        self.sequence_encoded_ = self.sequence_encoded_.copy()
        self.order_encoded_ = self.order_encoded_.copy()
        self.overlap_targets_ = self.overlap_targets_.copy()
        self.conditional_finger_targets_ = self.conditional_finger_targets_.copy()
        self.conditional_finger_active_ = self.conditional_finger_active_.copy()
        self.finger_encoded_[indices] = encoded_finger
        self.sequence_encoded_[indices] = encoded_sequence
        self.order_encoded_[indices] = encoded_order
        self.overlap_targets_[indices] = np.asarray(arrays[3], dtype=np.float32)
        self.conditional_finger_targets_[indices] = conditional
        self.conditional_finger_active_[indices] = active
        self.registered_calibration_indices_ = indices.copy()
        return self

    def _set_stage_trainable(self, stage: str) -> tuple[str, ...]:
        for parameter in self.model_.parameters():
            parameter.requires_grad_(False)
        prefixes = [
            "adapter_",
            "target_",
            "finger_head",
            "press_head",
            "conditional_finger_head",
            "sequence_head",
            "order_head",
            "overlap_head",
            "final_norm",
        ]
        if stage in {"last_block", "full"}:
            prefixes.extend([f"temporal_blocks.{len(self.model_.temporal_blocks) - 1}", "pool_projection"])
        if stage == "full":
            for name, parameter in self.model_.named_parameters():
                parameter.requires_grad_(not name.startswith("source_"))
        else:
            for name, parameter in self.model_.named_parameters():
                if any(name.startswith(prefix) for prefix in prefixes):
                    parameter.requires_grad_(True)
        return tuple(name for name, parameter in self.model_.named_parameters() if parameter.requires_grad)

    def _l2sp_penalty(self):
        torch = _torch()
        result = torch.zeros((), dtype=torch.float32, device=self.device_)
        for name, parameter in self.model_.named_parameters():
            if parameter.requires_grad:
                result = result + torch.sum((parameter - self.source_state_[name].to(self.device_)) ** 2)
        return result

    def _adapt_stage(self, calibration_indices: np.ndarray, *, stage: str, steps: int, learning_rate: float) -> None:
        torch = _torch()
        trainable_names = self._set_stage_trainable(stage)
        parameters = [parameter for parameter in self.model_.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(self.weight_decay))
        rng = np.random.default_rng(int(self.random_state) + {"adapter": 11, "last_block": 23, "full": 37}[stage])
        batch_size = int(self.batch_size)
        target_probabilities = None
        replay_probabilities = None
        if bool(self.balanced_sampling):
            target_probabilities = balanced_window_sampling_weights(
                self.row_domains_[calibration_indices],
                self.trial_ids_[calibration_indices],
                self.finger_encoded_[calibration_indices],
            )
            replay_probabilities = self.source_sampling_weights_[self.source_indices_]
            replay_probabilities = replay_probabilities / replay_probabilities.sum()
        running_loss = 0.0
        for _step in range(int(steps)):
            target_indices = rng.choice(
                calibration_indices,
                size=min(batch_size, calibration_indices.size),
                replace=calibration_indices.size < batch_size,
                p=target_probabilities,
            )
            x, finger, sequence, order, overlap, conditional_finger, conditional_active = self._tensor_batch(target_indices)
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model_(self._augment(x), use_adapter=True)
            loss = self._loss(
                outputs,
                finger,
                sequence,
                order,
                overlap,
                conditional_finger,
                conditional_active,
            )
            if float(self.source_replay_weight) > 0.0:
                replay_indices = rng.choice(
                    self.source_indices_,
                    size=min(batch_size, self.source_indices_.size),
                    replace=self.source_indices_.size < batch_size,
                    p=replay_probabilities,
                )
                sx, sf, ss, so, sov, scf, sca = self._tensor_batch(replay_indices)
                source_outputs = self.model_(
                    self._augment(sx),
                    use_adapter=False,
                    source_adapter_ids=self._source_adapter_ids(replay_indices),
                )
                loss = loss + float(self.source_replay_weight) * self._loss(
                    source_outputs,
                    sf,
                    ss,
                    so,
                    sov,
                    scf,
                    sca,
                )
            if float(self.l2sp_weight) > 0.0:
                loss = loss + float(self.l2sp_weight) * self._l2sp_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        self.adaptation_history_.append(
            {
                "stage": stage,
                "steps": int(steps),
                "mean_training_loss": running_loss / max(1, int(steps)),
                "trainable_parameter_names": trainable_names,
            }
        )

    def adapt_target_indices(
        self,
        calibration_indices: Sequence[int] | np.ndarray,
        *,
        n_calibration_trials: int,
        mode: str,
    ) -> "TorchProgressiveTemporalWindowClassifier":
        if mode not in {"adapter_only", "progressive_full"}:
            raise ValueError("mode must be adapter_only or progressive_full")
        calibration_indices = np.asarray(calibration_indices, dtype=int).reshape(-1)
        if calibration_indices.size == 0:
            raise ValueError("calibration_indices must not be empty")
        if np.any(calibration_indices < 0) or np.any(calibration_indices >= self.window_store_.shape[0]):
            raise ValueError("calibration_indices are out of range")
        registered = getattr(self, "registered_calibration_indices_", None)
        if bool(self.hierarchical) and (
            registered is None
            or not np.array_equal(np.sort(calibration_indices), np.sort(registered))
        ):
            raise RuntimeError("Hierarchical target adaptation requires explicitly registered calibration labels")
        self.allowed_label_indices_ = np.unique(np.concatenate((self.source_indices_, calibration_indices)))
        if bool(self.subject_specific_normalization):
            target_domains = np.unique(self.row_domains_[calibration_indices])
            if target_domains.size != 1:
                raise ValueError("Target calibration rows must belong to exactly one subject")
            self.target_normalization_domain_ = target_domains[0]
            total = np.zeros(self.n_channels_, dtype=np.float64)
            squared = np.zeros_like(total)
            count = 0
            for start in range(0, calibration_indices.size, int(self.batch_size)):
                rows = calibration_indices[start : start + int(self.batch_size)]
                values = np.asarray(self.window_store_[rows], dtype=np.float32)
                total += values.sum(axis=(0, 1), dtype=np.float64)
                squared += np.square(values, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
                count += int(values.shape[0] * values.shape[1])
            self.target_sensor_mean_ = (total / count).astype(np.float32)
            variance = np.maximum(squared / count - np.square(total / count), 1e-12)
            self.target_sensor_std_ = np.sqrt(variance).astype(np.float32)
        self.model_.load_state_dict(self.source_state_)
        self.adaptation_history_ = []
        stages: list[tuple[str, int, float]] = [("adapter", int(self.adapter_steps), float(self.adaptation_learning_rate))]
        if mode == "progressive_full" and n_calibration_trials >= int(self.min_trials_for_last_block):
            stages.append(("last_block", int(self.last_block_steps), float(self.adaptation_learning_rate)))
        if mode == "progressive_full" and n_calibration_trials >= int(self.min_trials_for_full_finetune):
            stages.append(("full", int(self.full_finetune_steps), float(self.full_finetune_learning_rate)))
        for stage, steps, learning_rate in stages:
            if steps > 0:
                self._adapt_stage(calibration_indices, stage=stage, steps=steps, learning_rate=learning_rate)
        self.model_.eval()
        return self

    def predict_outputs_indices(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        source_domain_mode: bool = False,
    ) -> dict[str, np.ndarray]:
        torch = _torch()
        indices = np.asarray(indices, dtype=int).reshape(-1)
        rows: dict[str, list[np.ndarray]] = {}
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, indices.size, int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x = torch.as_tensor(self._normalized_batch(batch_indices), dtype=torch.float32, device=self.device_)
                outputs = self.model_(
                    x,
                    use_adapter=not source_domain_mode,
                    source_adapter_ids=self._source_adapter_ids(batch_indices) if source_domain_mode else None,
                )
                for name, values in outputs.items():
                    rows.setdefault(name, []).append(values.detach().cpu().numpy())
        result = {name: np.concatenate(values, axis=0) for name, values in rows.items()}
        result["finger_probabilities_auxiliary"] = torch.softmax(
            torch.as_tensor(result["finger"]), dim=1
        ).numpy()
        result["sequence_probabilities"] = torch.softmax(
            torch.as_tensor(result["sequence"]), dim=1
        ).numpy()
        result["order_probabilities"] = torch.softmax(
            torch.as_tensor(result["order"]), dim=1
        ).numpy()
        if bool(self.hierarchical):
            result["press_probabilities"] = torch.softmax(torch.as_tensor(result["press"]), dim=1).numpy()
            result["conditional_finger_probabilities"] = torch.softmax(
                torch.as_tensor(result["conditional_finger"]), dim=1
            ).numpy()
            result["probabilities"] = combine_hierarchical_probabilities(
                result["press_probabilities"],
                result["conditional_finger_probabilities"],
            )
        else:
            result["probabilities"] = result["finger_probabilities_auxiliary"].astype(float, copy=False)
        return result

    def predict_proba_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        if bool(self.hierarchical):
            return self.predict_outputs_indices(indices)["probabilities"]
        torch = _torch()
        indices = np.asarray(indices, dtype=int).reshape(-1)
        rows: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, indices.size, int(self.batch_size)):
                batch_indices = indices[start : start + int(self.batch_size)]
                x = torch.as_tensor(self._normalized_batch(batch_indices), dtype=torch.float32, device=self.device_)
                outputs = self.model_(x, use_adapter=True)
                rows.append(torch.softmax(outputs["finger"], dim=1).detach().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(float, copy=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "progressive_temporal_window_multitask",
            "hidden_units": int(self.hidden_units),
            "num_blocks": int(self.num_blocks),
            "adapter_rank": int(self.adapter_rank),
            "source_epochs": int(self.source_epochs),
            "best_source_epoch": int(getattr(self, "best_source_epoch_", 0)),
            "best_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "source_validation_mode": getattr(self, "source_validation_mode_", "unknown"),
            "source_validation_domain": getattr(self, "source_validation_domain_", None),
            "source_refit_all": bool(self.source_refit_all),
            "sequence_loss_weight": float(self.sequence_loss_weight),
            "order_loss_weight": float(self.order_loss_weight),
            "overlap_loss_weight": float(self.overlap_loss_weight),
            "hierarchical": bool(self.hierarchical),
            "balanced_sampling": bool(self.balanced_sampling),
            "subject_specific_normalization": bool(self.subject_specific_normalization),
            "source_specific_adapters": bool(self.source_specific_adapters),
            "source_selection_metric": str(self.source_selection_metric),
            "adapter_kind": str(self.adapter_kind),
            "best_source_validation_accuracy": float(getattr(self, "best_source_validation_accuracy_", np.nan)),
            "adaptation_history": copy.deepcopy(getattr(self, "adaptation_history_", [])),
        }


__all__ = ["TorchProgressiveTemporalWindowClassifier"]
