"""Physical-finger source pretraining for participant-local sequence decoding.

The Katja button-press task has five physical fingers, while each participant's
four scored classes are the four non-fixed fingers. Participant-local class
indices therefore need not denote the same physical finger across people. This
module keeps a globally meaningful physical-finger head during source training
and selects the four target-relevant head columns during labeled target
adaptation. Source replay continues to use the full physical head, while target
losses and predictions use the participant-local four-class view.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from neureptrace.decoding._progressive_sequence_core import (
    PROGRESSIVE_SEQUENCE_CATEGORY,
    _as_feature_tensor,
    _as_label_matrix,
    _as_object_vector,
    _bounded_float,
    _nonnegative_float,
    _nonnegative_int,
    _positive_float,
    _positive_int,
    _torch,
)
from neureptrace.decoding._progressive_sequence_model import TorchProgressiveSequenceClassifier
from neureptrace.decoding._progressive_sequence_network import (
    _source_trial_validation_split,
    _target_trial_validation_split,
    _torch_log_sinkhorn,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

PHYSICAL_FINGER_SEQUENCE_PROTOCOL = "physical_finger_progressive_sequence_target_fine_tuning"


def _PhysicalFingerSequenceModule(
    *,
    input_dim: int,
    n_events: int,
    n_physical_classes: int,
    hidden_units: int,
    num_layers: int,
    num_heads: int,
    feedforward_multiplier: float,
    adapter_rank: int,
    adapter_alpha: float,
    dropout: float,
):
    """Build the sequence network with a persistent global physical head."""

    torch = _torch()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = torch.nn.LayerNorm(input_dim)
            self.base_projection = torch.nn.Linear(input_dim, hidden_units)
            self.adapter_down = torch.nn.Linear(input_dim, adapter_rank, bias=False)
            self.adapter_up = torch.nn.Linear(adapter_rank, hidden_units, bias=False)
            self.position_embedding = torch.nn.Parameter(torch.zeros(1, n_events, hidden_units))
            feedforward_units = max(hidden_units, int(round(hidden_units * feedforward_multiplier)))
            self.event_mlp = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_units),
                torch.nn.Linear(hidden_units, feedforward_units),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(feedforward_units, hidden_units),
            )
            self.sequence_layers = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_units,
                        nhead=num_heads,
                        dim_feedforward=feedforward_units,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_norm = torch.nn.LayerNorm(hidden_units)
            self.class_head = torch.nn.Linear(hidden_units, n_physical_classes)
            self.adapter_scale = float(adapter_alpha) / float(adapter_rank)
            self.register_buffer(
                "target_class_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            torch.nn.init.zeros_(self.adapter_up.weight)
            torch.nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        def set_target_class_indices(self, indices) -> None:
            tensor = torch.as_tensor(indices, dtype=torch.long, device=self.position_embedding.device)
            if tensor.ndim != 1 or tensor.numel() != n_events:
                raise ValueError(f"target physical class indices must contain exactly {n_events} entries.")
            if torch.unique(tensor).numel() != tensor.numel():
                raise ValueError("target physical class indices must be unique.")
            if torch.any(tensor < 0) or torch.any(tensor >= n_physical_classes):
                raise ValueError("target physical class index is outside the source head.")
            self.target_class_indices = tensor

        def clear_target_class_indices(self) -> None:
            self.target_class_indices = torch.empty(
                0,
                dtype=torch.long,
                device=self.position_embedding.device,
            )

        def forward(self, features, *, use_adapter: bool = True):
            normalized = self.input_norm(features)
            hidden = self.base_projection(normalized)
            if use_adapter:
                hidden = hidden + self.adapter_scale * self.adapter_up(self.adapter_down(normalized))
            hidden = hidden + self.position_embedding
            hidden = hidden + self.event_mlp(hidden)
            for layer in self.sequence_layers:
                hidden = layer(hidden)
            logits = self.class_head(self.final_norm(hidden))
            if use_adapter and self.target_class_indices.numel():
                return logits.index_select(2, self.target_class_indices)
            return logits

    return Module()


def _encode_with_classes(labels: np.ndarray, classes: np.ndarray, *, name: str) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes.tolist())}
    encoded = np.empty(labels.shape, dtype=np.int64)
    for index in np.ndindex(labels.shape):
        label = labels[index]
        if label not in mapping:
            raise ValueError(f"{name} contains unknown class {label!r}.")
        encoded[index] = mapping[label]
    return encoded


class TorchPhysicalFingerSequenceClassifier(TorchProgressiveSequenceClassifier):
    """Pretrain on global physical fingers and adapt to four local classes.

    The source model has one output per observed physical finger. Once target
    calibration is supplied, each local target class is matched to its unique
    physical finger using calibration rows only. Target adaptation sees the
    corresponding four-column view of the physical head; source replay still
    sees the complete physical head.
    """

    def fit_source(
        self,
        source_features: Sequence | np.ndarray,
        source_physical_labels: Sequence | np.ndarray,
        *,
        source_subjects: Sequence[Any] | np.ndarray | None = None,
    ) -> "TorchPhysicalFingerSequenceClassifier":
        torch = _torch()
        x = _as_feature_tensor(source_features, name="source_features")
        labels = _as_label_matrix(source_physical_labels, name="source_physical_labels")
        if labels.shape != x.shape[:2]:
            raise ValueError("source_physical_labels must match the source trial and event dimensions.")
        self.physical_classes_, inverse = np.unique(labels, return_inverse=True)
        y = inverse.reshape(labels.shape).astype(np.int64, copy=False)
        self.n_events_ = int(x.shape[1])
        self.n_features_in_ = int(x.shape[2])
        self.n_physical_classes_ = int(self.physical_classes_.shape[0])
        if self.n_physical_classes_ < self.n_events_:
            raise ValueError("Physical-finger source training needs at least one physical class per scored event.")
        if any(np.unique(row).shape[0] != self.n_events_ for row in y):
            raise ValueError("Every source trial must contain distinct physical fingers.")

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
        self.model_ = _PhysicalFingerSequenceModule(
            input_dim=self.n_features_in_,
            n_events=self.n_events_,
            n_physical_classes=self.n_physical_classes_,
            hidden_units=hidden_units,
            num_layers=_positive_int(self.num_layers, "num_layers"),
            num_heads=num_heads,
            feedforward_multiplier=_positive_float(self.feedforward_multiplier, "feedforward_multiplier"),
            adapter_rank=_positive_int(self.adapter_rank, "adapter_rank"),
            adapter_alpha=_positive_float(self.adapter_alpha, "adapter_alpha"),
            dropout=_bounded_float(self.dropout, "dropout", lower=0.0, upper=1.0),
        ).to(self.device_)
        self.model_.clear_target_class_indices()
        self.classes_ = self.physical_classes_.copy()
        self.n_classes_ = self.n_physical_classes_
        self._set_source_trainable()
        train_idx, validation_idx, validation_mode = _source_trial_validation_split(
            y,
            domains,
            validation_fraction=_bounded_float(
                self.validation_fraction,
                "validation_fraction",
                lower=0.0,
                upper=1.0,
            ),
            random_state=seed,
        )
        self._train_source(x, y, domains, train_idx=train_idx, validation_idx=validation_idx)
        self.source_validation_mode_ = validation_mode
        self.meta_episodes_run_, self.meta_episodes_accepted_ = self._meta_initialize_target_adapter(x, y, domains)
        self.source_features_ = x.copy()
        self.source_encoded_labels_ = y.copy()
        self.source_domains_encoded_ = domains.copy()
        self.source_state_ = {
            name: value.detach().cpu().clone()
            for name, value in self.model_.state_dict().items()
        }
        self.model_.eval()
        return self

    def adapt_target(
        self,
        target_calibration_features: Sequence | np.ndarray,
        target_calibration_labels: Sequence | np.ndarray,
        *,
        target_calibration_physical_labels: Sequence | np.ndarray,
        target_strata: Sequence[Any] | np.ndarray | None = None,
    ) -> "TorchPhysicalFingerSequenceClassifier":
        if not hasattr(self, "source_state_"):
            raise RuntimeError("fit_source must be called before adapt_target.")
        x = _as_feature_tensor(target_calibration_features, name="target_calibration_features")
        labels = _as_label_matrix(target_calibration_labels, name="target_calibration_labels")
        physical = _as_label_matrix(
            target_calibration_physical_labels,
            name="target_calibration_physical_labels",
        )
        if x.shape[1:] != (self.n_events_, self.n_features_in_):
            raise ValueError("Target calibration features must match the fitted source event and feature dimensions.")
        if labels.shape != x.shape[:2] or physical.shape != labels.shape:
            raise ValueError("Target local and physical labels must match target calibration features.")

        target_classes = np.unique(labels)
        if target_classes.shape[0] != self.n_events_:
            raise ValueError(f"Target calibration must contain exactly {self.n_events_} participant-local classes.")
        y = _encode_with_classes(labels, target_classes, name="target_calibration_labels")

        physical_to_index = {label: index for index, label in enumerate(self.physical_classes_.tolist())}
        target_physical_codes: list[Any] = []
        target_physical_indices: list[int] = []
        for target_class in target_classes.tolist():
            class_codes = np.unique(physical[labels == target_class])
            if class_codes.shape[0] != 1:
                raise ValueError(f"Target class {target_class!r} must map to exactly one physical finger in calibration rows.")
            physical_code = class_codes[0]
            if physical_code not in physical_to_index:
                raise ValueError(f"Target physical finger {physical_code!r} is absent from source physical classes.")
            target_physical_codes.append(physical_code)
            target_physical_indices.append(physical_to_index[physical_code])
        if len(set(target_physical_indices)) != self.n_events_:
            raise ValueError("Target local classes must map to distinct physical fingers.")
        expected_physical = set(target_physical_codes)
        for trial_index, (trial_local, trial_physical) in enumerate(zip(labels, physical, strict=True)):
            if set(trial_physical.tolist()) != expected_physical:
                raise ValueError(
                    f"Target calibration trial {trial_index} does not contain each mapped physical finger exactly once."
                )
            for local_label, physical_label in zip(trial_local.tolist(), trial_physical.tolist(), strict=True):
                local_index = int(np.flatnonzero(target_classes == local_label)[0])
                if physical_label != target_physical_codes[local_index]:
                    raise ValueError("Target local-to-physical mapping is inconsistent across calibration trials.")

        self.model_.load_state_dict(self.source_state_)
        self.model_.set_target_class_indices(target_physical_indices)
        self.target_classes_ = target_classes.copy()
        self.target_physical_codes_ = np.asarray(target_physical_codes)
        self.target_physical_indices_ = np.asarray(target_physical_indices, dtype=int)
        self.classes_ = self.target_classes_.copy()
        self.n_classes_ = int(self.target_classes_.shape[0])
        self._validate_permutation_labels(y, name="target_calibration_labels")

        strata = None if target_strata is None else _as_object_vector(target_strata, name="target_strata")
        if strata is not None and strata.shape[0] != x.shape[0]:
            raise ValueError("target_strata must contain one value per target calibration trial.")
        reset_rng = getattr(self, "_reset_torch_rng", None)
        if callable(reset_rng):
            reset_rng("physical_target_adaptation", x.shape[0], *target_physical_codes)
        train_idx, validation_idx, validation_mode = _target_trial_validation_split(
            x.shape[0],
            strata=strata,
            validation_fraction=_bounded_float(
                self.validation_fraction,
                "validation_fraction",
                lower=0.0,
                upper=1.0,
            ),
            random_state=self.random_state,
        )
        self.target_validation_mode_ = validation_mode
        self.adaptation_stage_history_ = []
        stages: list[tuple[str, int, float]] = [
            (
                "adapter",
                _nonnegative_int(self.adapter_steps, "adapter_steps"),
                _positive_float(self.adaptation_learning_rate, "adaptation_learning_rate"),
            )
        ]
        if x.shape[0] >= _positive_int(self.min_trials_for_last_block, "min_trials_for_last_block"):
            stages.append(
                (
                    "last_block",
                    _nonnegative_int(self.last_block_steps, "last_block_steps"),
                    _positive_float(self.adaptation_learning_rate, "adaptation_learning_rate"),
                )
            )
        if x.shape[0] >= _positive_int(self.min_trials_for_full_finetune, "min_trials_for_full_finetune"):
            stages.append(
                (
                    "full",
                    _nonnegative_int(self.full_finetune_steps, "full_finetune_steps"),
                    _positive_float(self.full_finetune_learning_rate, "full_finetune_learning_rate"),
                )
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

    def _classification_loss(self, logits, labels, *, domains=None, include_vrex: bool):
        """Use physical CE for source logits and structured local CE for target logits."""

        torch = _torch()
        output_classes = int(logits.shape[-1])
        event_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, output_classes),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        trial_losses = event_losses.mean(dim=1)
        if domains is None:
            loss = trial_losses.mean()
        else:
            group_losses = torch.stack(
                [trial_losses[domains == domain].mean() for domain in torch.unique(domains)]
            )
            loss = group_losses.mean()
            if include_vrex and group_losses.numel() > 1:
                loss = loss + _nonnegative_float(
                    self.source_vrex_weight,
                    "source_vrex_weight",
                ) * group_losses.var(unbiased=False)

        target_view = output_classes == self.n_events_
        if self.enforce_permutation_labels and target_view:
            sinkhorn_weight = _nonnegative_float(self.sinkhorn_loss_weight, "sinkhorn_loss_weight")
            assignment_weight = _nonnegative_float(self.assignment_loss_weight, "assignment_loss_weight")
            if sinkhorn_weight > 0.0:
                log_assignment = _torch_log_sinkhorn(
                    logits,
                    temperature=_positive_float(self.sinkhorn_temperature, "sinkhorn_temperature"),
                    iterations=_positive_int(self.sinkhorn_iterations, "sinkhorn_iterations"),
                )
                selected = log_assignment.gather(2, labels.unsqueeze(2)).squeeze(2)
                loss = loss + sinkhorn_weight * (-selected.mean())
            if assignment_weight > 0.0:
                class_mass = torch.softmax(logits, dim=2).sum(dim=1)
                loss = loss + assignment_weight * torch.mean((class_mass - 1.0) ** 2)
        return loss

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "progressive_sequence_protocol": PHYSICAL_FINGER_SEQUENCE_PROTOCOL,
                "progressive_sequence_protocol_category": PROGRESSIVE_SEQUENCE_CATEGORY,
                "physical_finger_source_pretraining": True,
                "physical_finger_source_classes": tuple(
                    getattr(self, "physical_classes_", np.asarray([])).tolist()
                ),
                "physical_finger_target_codes": tuple(
                    getattr(self, "target_physical_codes_", np.asarray([])).tolist()
                ),
                "physical_finger_target_head_indices": tuple(
                    int(value)
                    for value in getattr(
                        self,
                        "target_physical_indices_",
                        np.asarray([], dtype=int),
                    ).tolist()
                ),
            }
        )
        return metadata
