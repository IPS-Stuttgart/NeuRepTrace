"""Training internals for the progressive sequence classifier."""

from __future__ import annotations

import copy
import numpy as np

from neureptrace.decoding._progressive_sequence_core import (
    _bounded_float,
    _nonnegative_float,
    _nonnegative_int,
    _positive_float,
    _positive_int,
    _stable_seed,
    _torch,
)
from neureptrace.decoding._progressive_sequence_network import (
    _epoch_batches,
    _set_model_stage_trainable,
    _torch_log_sinkhorn,
)


class _ProgressiveSequenceTrainingMixin:
    def _resolve_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _encode_target_labels(self, labels: np.ndarray) -> np.ndarray:
        mapping = {label: index for index, label in enumerate(self.classes_.tolist())}
        encoded = np.empty(labels.shape, dtype=np.int64)
        for index in np.ndindex(labels.shape):
            label = labels[index]
            if label not in mapping:
                raise ValueError(f"target_calibration_labels contains class {label!r}, absent from source classes.")
            encoded[index] = mapping[label]
        return encoded

    def _validate_permutation_labels(self, encoded_labels: np.ndarray, *, name: str) -> None:
        if not self.enforce_permutation_labels:
            return
        if self.n_events_ != self.n_classes_:
            raise ValueError("Permutation-constrained training requires n_events == n_classes.")
        expected = np.arange(self.n_classes_)
        if not np.all(np.sort(encoded_labels, axis=1) == expected[None, :]):
            raise ValueError(f"Every {name} trial must contain one occurrence of every class.")

    def _set_source_trainable(self) -> None:
        for name, parameter in self.model_.named_parameters():
            parameter.requires_grad_(not name.startswith("adapter_"))

    def _set_adaptation_stage_trainable(self, stage: str) -> tuple[str, ...]:
        return _set_model_stage_trainable(self.model_, stage)

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
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model_.parameters() if parameter.requires_grad],
            lr=_positive_float(self.source_learning_rate, "source_learning_rate"),
            weight_decay=_nonnegative_float(self.weight_decay, "weight_decay"),
        )
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device_)
        domain_tensor = torch.as_tensor(domains, dtype=torch.long, device=self.device_)
        rng = np.random.default_rng(self.random_state)
        batch_size = _positive_int(self.batch_size, "batch_size")
        max_epochs = _positive_int(self.source_max_epochs, "source_max_epochs")
        patience_limit = _positive_int(self.patience, "patience")
        best_loss = np.inf
        best_state = None
        patience_left = patience_limit
        self.source_epochs_run_ = 0
        for epoch in range(max_epochs):
            self.source_epochs_run_ = epoch + 1
            self.model_.train()
            for batch_idx in _epoch_batches(train_idx, batch_size=batch_size, rng=rng):
                optimizer.zero_grad(set_to_none=True)
                batch_x = self._augment(x_tensor[batch_idx])
                logits = self.model_(batch_x, use_adapter=False)
                loss = self._classification_loss(logits, y_tensor[batch_idx], domains=domain_tensor[batch_idx], include_vrex=True)
                loss.backward()
                optimizer.step()
            validation_loss = self._validation_loss(x_tensor[validation_idx], y_tensor[validation_idx], use_adapter=False)
            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
                patience_left = patience_limit
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_source_validation_loss_ = float(best_loss)

    def _meta_initialize_target_adapter(self, x: np.ndarray, y: np.ndarray, domains: np.ndarray) -> tuple[int, int]:
        """Reptile-style source-subject episodes for the target adapter initialization."""

        torch = _torch()
        epochs = _nonnegative_int(self.meta_epochs, "meta_epochs")
        if epochs == 0 or np.unique(domains).size < 2:
            return 0, 0
        support_trials = _positive_int(self.meta_support_trials, "meta_support_trials")
        query_trials = _positive_int(self.meta_query_trials, "meta_query_trials")
        inner_steps = _positive_int(self.meta_inner_steps, "meta_inner_steps")
        learning_rate = _positive_float(self.meta_learning_rate, "meta_learning_rate")
        step_size = _bounded_float(self.meta_step_size, "meta_step_size", lower=0.0, upper=1.0)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device_)
        rng = np.random.default_rng(_stable_seed(_nonnegative_int(self.random_state or 0, "random_state"), (), "meta"))
        episodes_run = 0
        episodes_accepted = 0
        domain_values = np.unique(domains)
        for _epoch in range(epochs):
            for domain in rng.permutation(domain_values):
                positions = np.flatnonzero(domains == domain)
                required = support_trials + query_trials
                if positions.size < required:
                    continue
                selected = rng.permutation(positions)[:required]
                support_idx = selected[:support_trials]
                query_idx = selected[support_trials:]
                clone = copy.deepcopy(self.model_).to(self.device_)
                _set_model_stage_trainable(clone, "adapter")
                optimizer = torch.optim.AdamW(
                    [parameter for parameter in clone.parameters() if parameter.requires_grad],
                    lr=learning_rate,
                    weight_decay=_nonnegative_float(self.weight_decay, "weight_decay"),
                )
                self.model_.eval()
                with torch.no_grad():
                    before_logits = self.model_(x_tensor[query_idx], use_adapter=True)
                    before_loss = float(
                        self._classification_loss(before_logits, y_tensor[query_idx], domains=None, include_vrex=False).detach().cpu()
                    )
                for _inner in range(inner_steps):
                    clone.train()
                    optimizer.zero_grad(set_to_none=True)
                    support_logits = clone(self._augment(x_tensor[support_idx]), use_adapter=True)
                    support_loss = self._classification_loss(support_logits, y_tensor[support_idx], domains=None, include_vrex=False)
                    support_loss.backward()
                    optimizer.step()
                clone.eval()
                with torch.no_grad():
                    after_logits = clone(x_tensor[query_idx], use_adapter=True)
                    after_loss = float(
                        self._classification_loss(after_logits, y_tensor[query_idx], domains=None, include_vrex=False).detach().cpu()
                    )
                episodes_run += 1
                if not np.isfinite(after_loss) or after_loss > before_loss + 1e-6:
                    continue
                clone_parameters = dict(clone.named_parameters())
                with torch.no_grad():
                    for name, parameter in self.model_.named_parameters():
                        if name.startswith("adapter_"):
                            parameter.add_(step_size * (clone_parameters[name].detach() - parameter))
                episodes_accepted += 1
        self.model_.eval()
        return int(episodes_run), int(episodes_accepted)

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
        trainable_names = self._set_adaptation_stage_trainable(stage)
        trainable_parameters = [parameter for parameter in self.model_.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            return
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=_nonnegative_float(self.weight_decay, "weight_decay"),
        )
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device_)
        source_x_tensor = torch.as_tensor(self.source_features_, dtype=torch.float32, device=self.device_)
        source_y_tensor = torch.as_tensor(self.source_encoded_labels_, dtype=torch.long, device=self.device_)
        rng = np.random.default_rng(_stable_seed(_nonnegative_int(self.random_state or 0, "random_state"), (), "stage", stage))
        batch_size = _positive_int(self.batch_size, "batch_size")
        replay_weight = _nonnegative_float(self.source_replay_weight, "source_replay_weight")
        l2sp_weight = _nonnegative_float(self.l2sp_weight, "l2sp_weight")
        patience_limit = _positive_int(self.patience, "patience")
        patience_left = patience_limit
        best_loss = np.inf
        best_state = None
        steps_run = 0
        for step in range(steps):
            steps_run = step + 1
            self.model_.train()
            target_batch = rng.choice(train_idx, size=min(batch_size, train_idx.size), replace=train_idx.size < batch_size)
            optimizer.zero_grad(set_to_none=True)
            target_logits = self.model_(self._augment(x_tensor[target_batch]), use_adapter=True)
            loss = self._classification_loss(target_logits, y_tensor[target_batch], domains=None, include_vrex=False)
            if replay_weight > 0.0:
                replay_size = min(batch_size, self.source_features_.shape[0])
                source_batch = rng.choice(self.source_features_.shape[0], size=replay_size, replace=self.source_features_.shape[0] < replay_size)
                source_logits = self.model_(self._augment(source_x_tensor[source_batch]), use_adapter=False)
                replay_loss = self._classification_loss(source_logits, source_y_tensor[source_batch], domains=None, include_vrex=False)
                loss = loss + replay_weight * replay_loss
            if l2sp_weight > 0.0:
                loss = loss + l2sp_weight * self._l2sp_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
            optimizer.step()

            validation_loss = self._validation_loss(x_tensor[validation_idx], y_tensor[validation_idx], use_adapter=True)
            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in self.model_.state_dict().items()}
                patience_left = patience_limit
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.adaptation_stage_history_.append(
            {
                "stage": stage,
                "steps_run": int(steps_run),
                "best_validation_loss": float(best_loss),
                "trainable_parameter_names": trainable_names,
            }
        )

    def _classification_loss(self, logits, labels, *, domains=None, include_vrex: bool):
        torch = _torch()
        event_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.n_classes_),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        trial_losses = event_losses.mean(dim=1)
        if domains is None:
            loss = trial_losses.mean()
        else:
            group_losses = torch.stack([trial_losses[domains == domain].mean() for domain in torch.unique(domains)])
            loss = group_losses.mean()
            if include_vrex and group_losses.numel() > 1:
                loss = loss + _nonnegative_float(self.source_vrex_weight, "source_vrex_weight") * group_losses.var(unbiased=False)

        if self.enforce_permutation_labels:
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

    def _validation_loss(self, features, labels, *, use_adapter: bool) -> float:
        self.model_.eval()
        torch = _torch()
        with torch.no_grad():
            logits = self.model_(features, use_adapter=use_adapter)
            loss = self._classification_loss(logits, labels, domains=None, include_vrex=False)
        return float(loss.detach().cpu())

    def _augment(self, features):
        torch = _torch()
        result = features
        noise_std = _nonnegative_float(self.feature_noise_std, "feature_noise_std")
        if noise_std > 0.0:
            result = result + torch.randn_like(result) * noise_std
        dropout = _bounded_float(self.feature_dropout, "feature_dropout", lower=0.0, upper=1.0)
        if dropout > 0.0:
            result = torch.nn.functional.dropout(result, p=dropout, training=True)
        return result

    def _l2sp_penalty(self):
        torch = _torch()
        penalties = []
        for name, parameter in self.model_.named_parameters():
            if not parameter.requires_grad or name.startswith("adapter_") or name not in self.source_state_:
                continue
            reference = self.source_state_[name].to(device=parameter.device, dtype=parameter.dtype)
            penalties.append(torch.mean((parameter - reference) ** 2))
        if not penalties:
            return torch.zeros((), dtype=torch.float32, device=self.device_)
        return torch.stack(penalties).mean()
