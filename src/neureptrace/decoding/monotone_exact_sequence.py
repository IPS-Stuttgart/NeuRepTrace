"""Monotone progressive adaptation with exact permutation supervision."""

from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding._progressive_sequence_core import (
    _nonnegative_float,
    _nonnegative_int,
    _positive_float,
    _positive_int,
    _stable_seed,
    _torch,
)
from neureptrace.decoding._progressive_sequence_network import _epoch_batches
from neureptrace.decoding._progressive_sequence_model import (
    TorchProgressiveSequenceClassifier,
)
from neureptrace.decoding.exact_permutation import torch_exact_permutation_nll

MONOTONE_EXACT_SEQUENCE_PROTOCOL = "progressive_sequence_monotone_exact_permutation"


def _state_copy(model) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


class TorchMonotoneExactSequenceClassifier(TorchProgressiveSequenceClassifier):
    """Progressive classifier that never selects a validation-worsening stage.

    Source epochs and target adaptation steps are selected on the existing
    leakage-free validation splits. Optionally, the selected training duration is
    replayed from the stage's incoming state on all available training rows. An
    exact permutation negative log-likelihood can replace the approximate
    Sinkhorn objective for small square trial assignments.
    """

    def __init__(
        self,
        *args: Any,
        exact_permutation_loss_weight: float = 0.0,
        exact_permutation_temperature: float = 1.0,
        refit_source_on_all: bool = True,
        refit_target_on_all: bool = True,
        selection_tolerance: float = 1e-6,
        **kwargs: Any,
    ):
        self.exact_permutation_loss_weight = _nonnegative_float(
            exact_permutation_loss_weight,
            "exact_permutation_loss_weight",
        )
        self.exact_permutation_temperature = _positive_float(
            exact_permutation_temperature,
            "exact_permutation_temperature",
        )
        self.refit_source_on_all = bool(refit_source_on_all)
        self.refit_target_on_all = bool(refit_target_on_all)
        self.selection_tolerance = _nonnegative_float(
            selection_tolerance,
            "selection_tolerance",
        )
        super().__init__(*args, **kwargs)

    def _reset_torch_rng(self, *parts: Any) -> int:
        """Reset CPU and CUDA RNGs for order-independent stage fitting."""

        torch = _torch()
        seed = _stable_seed(
            _nonnegative_int(self.random_state or 0, "random_state"),
            (),
            *parts,
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return int(seed)

    def _source_epoch_updates(
        self,
        *,
        x_tensor,
        y_tensor,
        domain_tensor,
        indices: np.ndarray,
        optimizer,
        rng: np.random.Generator,
        batch_size: int,
    ) -> None:
        self.model_.train()
        for batch_idx in _epoch_batches(
            indices,
            batch_size=batch_size,
            rng=rng,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model_(
                self._augment(x_tensor[batch_idx]),
                use_adapter=False,
            )
            loss = self._classification_loss(
                logits,
                y_tensor[batch_idx],
                domains=domain_tensor[batch_idx],
                include_vrex=True,
            )
            loss.backward()
            optimizer.step()

    def _train_source(
        self,
        x: np.ndarray,
        y: np.ndarray,
        domains: np.ndarray,
        *,
        train_idx: np.ndarray,
        validation_idx: np.ndarray,
    ) -> None:
        torch = _torch()
        initial_state = _state_copy(self.model_)
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model_.parameters()
                if parameter.requires_grad
            ],
            lr=_positive_float(
                self.source_learning_rate,
                "source_learning_rate",
            ),
            weight_decay=_nonnegative_float(
                self.weight_decay,
                "weight_decay",
            ),
        )
        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device_,
        )
        y_tensor = torch.as_tensor(
            y,
            dtype=torch.long,
            device=self.device_,
        )
        domain_tensor = torch.as_tensor(
            domains,
            dtype=torch.long,
            device=self.device_,
        )
        selection_seed = self._reset_torch_rng("source_selection")
        rng = np.random.default_rng(selection_seed)
        batch_size = _positive_int(self.batch_size, "batch_size")
        max_epochs = _positive_int(
            self.source_max_epochs,
            "source_max_epochs",
        )
        patience_limit = _positive_int(self.patience, "patience")
        best_loss = np.inf
        best_state = None
        best_epoch = 0
        patience_left = patience_limit
        self.source_epochs_run_ = 0
        for epoch in range(max_epochs):
            self.source_epochs_run_ = epoch + 1
            self._source_epoch_updates(
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                domain_tensor=domain_tensor,
                indices=train_idx,
                optimizer=optimizer,
                rng=rng,
                batch_size=batch_size,
            )
            validation_loss = self._validation_loss(
                x_tensor[validation_idx],
                y_tensor[validation_idx],
                use_adapter=False,
            )
            if validation_loss + self.selection_tolerance < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = _state_copy(self.model_)
                patience_left = patience_limit
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        all_indices = np.arange(x.shape[0], dtype=int)
        refit = (
            self.refit_source_on_all
            and best_epoch > 0
            and not np.array_equal(np.sort(train_idx), all_indices)
        )
        if refit:
            self.model_.load_state_dict(initial_state)
            self._set_source_trainable()
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in self.model_.parameters()
                    if parameter.requires_grad
                ],
                lr=_positive_float(
                    self.source_learning_rate,
                    "source_learning_rate",
                ),
                weight_decay=_nonnegative_float(
                    self.weight_decay,
                    "weight_decay",
                ),
            )
            refit_seed = self._reset_torch_rng(
                "source_refit",
                best_epoch,
            )
            rng = np.random.default_rng(refit_seed)
            for _epoch in range(best_epoch):
                self._source_epoch_updates(
                    x_tensor=x_tensor,
                    y_tensor=y_tensor,
                    domain_tensor=domain_tensor,
                    indices=all_indices,
                    optimizer=optimizer,
                    rng=rng,
                    batch_size=batch_size,
                )
            self.source_refit_epochs_ = int(best_epoch)
        else:
            if best_state is not None:
                self.model_.load_state_dict(best_state)
            self.source_refit_epochs_ = 0
        self.best_source_epoch_ = int(best_epoch)
        self.best_source_validation_loss_ = float(best_loss)

    def _adaptation_update(
        self,
        *,
        x_tensor,
        y_tensor,
        target_indices: np.ndarray,
        source_x_tensor,
        source_y_tensor,
        optimizer,
        trainable_parameters,
        rng: np.random.Generator,
        batch_size: int,
        replay_weight: float,
        l2sp_weight: float,
    ) -> None:
        self.model_.train()
        target_size = min(batch_size, target_indices.size)
        target_batch = rng.choice(
            target_indices,
            size=target_size,
            replace=False,
        )
        optimizer.zero_grad(set_to_none=True)
        target_logits = self.model_(
            self._augment(x_tensor[target_batch]),
            use_adapter=True,
        )
        loss = self._classification_loss(
            target_logits,
            y_tensor[target_batch],
            domains=None,
            include_vrex=False,
        )
        if replay_weight > 0.0:
            replay_size = min(batch_size, self.source_features_.shape[0])
            source_batch = rng.choice(
                self.source_features_.shape[0],
                size=replay_size,
                replace=False,
            )
            source_logits = self.model_(
                self._augment(source_x_tensor[source_batch]),
                use_adapter=False,
            )
            replay_loss = self._classification_loss(
                source_logits,
                source_y_tensor[source_batch],
                domains=None,
                include_vrex=False,
            )
            loss = loss + replay_weight * replay_loss
        if l2sp_weight > 0.0:
            loss = loss + l2sp_weight * self._l2sp_penalty()
        loss.backward()
        torch = _torch()
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=5.0,
        )
        optimizer.step()

    def _adapt_stage(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        train_idx: np.ndarray,
        validation_idx: np.ndarray,
        stage: str,
        steps: int,
        learning_rate: float,
    ) -> None:
        torch = _torch()
        stage_start_state = _state_copy(self.model_)
        trainable_names = self._set_adaptation_stage_trainable(stage)
        trainable_parameters = [
            parameter
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            return
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=_nonnegative_float(
                self.weight_decay,
                "weight_decay",
            ),
        )
        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device_,
        )
        y_tensor = torch.as_tensor(
            y,
            dtype=torch.long,
            device=self.device_,
        )
        source_x_tensor = torch.as_tensor(
            self.source_features_,
            dtype=torch.float32,
            device=self.device_,
        )
        source_y_tensor = torch.as_tensor(
            self.source_encoded_labels_,
            dtype=torch.long,
            device=self.device_,
        )
        selection_seed = self._reset_torch_rng(
            "stage_selection",
            stage,
            x.shape[0],
        )
        rng = np.random.default_rng(selection_seed)
        batch_size = _positive_int(self.batch_size, "batch_size")
        replay_weight = _nonnegative_float(
            self.source_replay_weight,
            "source_replay_weight",
        )
        l2sp_weight = _nonnegative_float(
            self.l2sp_weight,
            "l2sp_weight",
        )
        patience_limit = _positive_int(self.patience, "patience")
        initial_validation_loss = self._validation_loss(
            x_tensor[validation_idx],
            y_tensor[validation_idx],
            use_adapter=True,
        )
        best_loss = initial_validation_loss
        best_state = stage_start_state
        best_step = 0
        patience_left = patience_limit
        steps_run = 0
        for step in range(steps):
            steps_run = step + 1
            self._adaptation_update(
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                target_indices=train_idx,
                source_x_tensor=source_x_tensor,
                source_y_tensor=source_y_tensor,
                optimizer=optimizer,
                trainable_parameters=trainable_parameters,
                rng=rng,
                batch_size=batch_size,
                replay_weight=replay_weight,
                l2sp_weight=l2sp_weight,
            )
            validation_loss = self._validation_loss(
                x_tensor[validation_idx],
                y_tensor[validation_idx],
                use_adapter=True,
            )
            if validation_loss + self.selection_tolerance < best_loss:
                best_loss = validation_loss
                best_step = step + 1
                best_state = _state_copy(self.model_)
                patience_left = patience_limit
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        all_indices = np.arange(x.shape[0], dtype=int)
        refit = (
            self.refit_target_on_all
            and best_step > 0
            and not np.array_equal(np.sort(train_idx), all_indices)
        )
        if refit:
            self.model_.load_state_dict(stage_start_state)
            trainable_names = self._set_adaptation_stage_trainable(stage)
            trainable_parameters = [
                parameter
                for parameter in self.model_.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=learning_rate,
                weight_decay=_nonnegative_float(
                    self.weight_decay,
                    "weight_decay",
                ),
            )
            refit_seed = self._reset_torch_rng(
                "stage_refit",
                stage,
                x.shape[0],
                best_step,
            )
            rng = np.random.default_rng(refit_seed)
            for _step in range(best_step):
                self._adaptation_update(
                    x_tensor=x_tensor,
                    y_tensor=y_tensor,
                    target_indices=all_indices,
                    source_x_tensor=source_x_tensor,
                    source_y_tensor=source_y_tensor,
                    optimizer=optimizer,
                    trainable_parameters=trainable_parameters,
                    rng=rng,
                    batch_size=batch_size,
                    replay_weight=replay_weight,
                    l2sp_weight=l2sp_weight,
                )
        else:
            self.model_.load_state_dict(best_state)

        self.adaptation_stage_history_.append(
            {
                "stage": stage,
                "steps_run": int(steps_run),
                "best_step": int(best_step),
                "stage_accepted": bool(best_step > 0),
                "initial_validation_loss": float(initial_validation_loss),
                "best_validation_loss": float(best_loss),
                "refit_steps": int(best_step if refit else 0),
                "refit_on_all_calibration": bool(refit),
                "trainable_parameter_names": trainable_names,
            }
        )

    def _classification_loss(
        self,
        logits,
        labels,
        *,
        domains=None,
        include_vrex: bool,
    ):
        loss = super()._classification_loss(
            logits,
            labels,
            domains=domains,
            include_vrex=include_vrex,
        )
        if self.exact_permutation_loss_weight > 0.0:
            if not self.enforce_permutation_labels:
                raise ValueError(
                    "exact_permutation_loss_weight requires "
                    "enforce_permutation_labels=True."
                )
            loss = loss + self.exact_permutation_loss_weight * (
                torch_exact_permutation_nll(
                    logits,
                    labels,
                    temperature=self.exact_permutation_temperature,
                )
            )
        return loss

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "monotone_exact_sequence_protocol": MONOTONE_EXACT_SEQUENCE_PROTOCOL,
                "monotone_stage_selection": True,
                "selection_tolerance": float(self.selection_tolerance),
                "refit_source_on_all": bool(self.refit_source_on_all),
                "refit_target_on_all": bool(self.refit_target_on_all),
                "source_refit_epochs": int(
                    getattr(self, "source_refit_epochs_", 0)
                ),
                "best_source_epoch": int(getattr(self, "best_source_epoch_", 0)),
                "exact_permutation_loss_weight": float(
                    self.exact_permutation_loss_weight
                ),
                "exact_permutation_temperature": float(
                    self.exact_permutation_temperature
                ),
                "uses_evaluation_labels": False,
            }
        )
        return metadata


__all__ = (
    "MONOTONE_EXACT_SEQUENCE_PROTOCOL",
    "TorchMonotoneExactSequenceClassifier",
)
