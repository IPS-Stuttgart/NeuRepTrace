"""Neural network and split helpers for progressive sequence adaptation."""

from __future__ import annotations

import numpy as np

from neureptrace.decoding._progressive_sequence_core import (
    _labels_equal,
    _nonnegative_int,
    _stable_seed,
    _torch,
    _unique_in_order,
)


def _set_model_stage_trainable(model, stage: str) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = ["adapter_", "class_head", "final_norm"]
    if stage in {"last_block", "full"}:
        prefixes.extend(["event_mlp", f"sequence_layers.{len(model.sequence_layers) - 1}"])
    if stage == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad_(True)
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def _SequenceAdapterModule(
    *,
    input_dim: int,
    n_events: int,
    n_classes: int,
    hidden_units: int,
    num_layers: int,
    num_heads: int,
    feedforward_multiplier: float,
    adapter_rank: int,
    adapter_alpha: float,
    dropout: float,
):
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
            self.class_head = torch.nn.Linear(hidden_units, n_classes)
            self.adapter_scale = float(adapter_alpha) / float(adapter_rank)
            torch.nn.init.zeros_(self.adapter_up.weight)
            torch.nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        def forward(self, features, *, use_adapter: bool = True):
            normalized = self.input_norm(features)
            hidden = self.base_projection(normalized)
            if use_adapter:
                hidden = hidden + self.adapter_scale * self.adapter_up(self.adapter_down(normalized))
            hidden = hidden + self.position_embedding
            hidden = hidden + self.event_mlp(hidden)
            for layer in self.sequence_layers:
                hidden = layer(hidden)
            return self.class_head(self.final_norm(hidden))

    return Module()


def _torch_log_sinkhorn(logits, *, temperature: float, iterations: int):
    torch = _torch()
    log_assignment = logits / float(temperature)
    for _ in range(iterations):
        log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=2, keepdim=True)
        log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=1, keepdim=True)
    return log_assignment - torch.logsumexp(log_assignment, dim=2, keepdim=True)


def _epoch_batches(indices: np.ndarray, *, batch_size: int, rng: np.random.Generator):
    shuffled = rng.permutation(indices)
    for start in range(0, shuffled.shape[0], batch_size):
        yield shuffled[start : start + batch_size]


def _source_trial_validation_split(
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(labels.shape[0], dtype=int)
    unique_domains = np.unique(domains)
    rng = np.random.default_rng(random_state)
    if 0.0 < validation_fraction < 1.0 and unique_domains.size >= 2:
        n_validation_domains = min(unique_domains.size - 1, max(1, int(round(unique_domains.size * validation_fraction))))
        validation_domains = rng.choice(unique_domains, size=n_validation_domains, replace=False)
        validation_mask = np.isin(domains, validation_domains)
        train_idx = indices[~validation_mask]
        validation_idx = indices[validation_mask]
        if train_idx.size and validation_idx.size:
            return train_idx, validation_idx, "heldout_source_subject"
    if 0.0 < validation_fraction < 1.0 and indices.size >= 2:
        shuffled = rng.permutation(indices)
        n_validation = min(indices.size - 1, max(1, int(round(indices.size * validation_fraction))))
        validation_idx = np.sort(shuffled[:n_validation])
        train_idx = np.sort(shuffled[n_validation:])
        return train_idx, validation_idx, "trial_fallback"
    return indices, indices, "training_loss_fallback"


def _target_trial_validation_split(
    n_trials: int,
    *,
    strata: np.ndarray | None,
    validation_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(n_trials, dtype=int)
    if not (0.0 < validation_fraction < 1.0) or n_trials < 2:
        return indices, indices, "training_loss_fallback"
    rng = np.random.default_rng(random_state)
    if strata is not None:
        train_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []
        for stratum_position, stratum in enumerate(_unique_in_order(strata.tolist())):
            positions = np.asarray(
                [index for index, candidate in enumerate(strata.tolist()) if _labels_equal(candidate, stratum)],
                dtype=int,
            )
            if positions.size < 2:
                return indices, indices, "training_loss_fallback"
            shuffled = np.random.default_rng(_stable_seed(_nonnegative_int(random_state or 0, "random_state"), (), "target", stratum_position, stratum)).permutation(positions)
            n_validation = min(positions.size - 1, max(1, int(round(positions.size * validation_fraction))))
            validation_parts.append(shuffled[:n_validation])
            train_parts.append(shuffled[n_validation:])
        return (
            np.sort(np.concatenate(train_parts).astype(int, copy=False)),
            np.sort(np.concatenate(validation_parts).astype(int, copy=False)),
            "stratified_target_trial",
        )
    shuffled = rng.permutation(indices)
    n_validation = min(n_trials - 1, max(1, int(round(n_trials * validation_fraction))))
    return np.sort(shuffled[n_validation:]), np.sort(shuffled[:n_validation]), "target_trial_fallback"
