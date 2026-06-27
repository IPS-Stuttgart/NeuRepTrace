"""Source-only VREx domain-generalization classifier.

VREx minimizes the mean source-domain risk together with the variance of risk
across source subjects.  The estimator accepts source features, source labels,
and source-domain ids only; held-out target data are never used during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from neureptrace.decoding.dann import _bounded_float, _integer, _nonnegative_float, _positive_float, _positive_int, _torch
from neureptrace.decoding.source_domain_generalization import (
    _SourceDGModule,
    _balanced_weights,
    _encode_inputs,
    _source_domain_validation_split,
)

SOURCE_VREX_PROTOCOL = "source_only_vrex"
SOURCE_VREX_CATEGORY = "1_strict_source_only"


@dataclass(frozen=True, slots=True)
class SourceVRExFitResult:
    """Fitted VREx model, target probabilities, and provenance."""

    model: "TorchVRExClassifier"
    probabilities: np.ndarray
    metadata: dict[str, Any]


class TorchVRExClassifier(ClassifierMixin, BaseEstimator):
    """Neural variance-risk-extrapolation decoder for strict LOSO.

    Training minimizes mean classification risk across source domains plus a
    penalty on the variance of those domain risks.  ``fit`` has no target-feature
    or target-label argument, making the estimator a Protocol-1 method.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        embedding_dim: int = 32,
        max_epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        penalty_weight: float = 1.0,
        penalty_anneal_epochs: int = 10,
        validation_fraction: float = 0.1,
        patience: int = 10,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
        device: str = "auto",
    ):
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.penalty_weight = penalty_weight
        self.penalty_anneal_epochs = penalty_anneal_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.class_weight = class_weight
        self.device = device

    def _resolve_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def fit(self, source_features: np.ndarray, source_labels: np.ndarray, *, source_domains: np.ndarray):
        x, classes, y, domain_names, domains = _encode_inputs(
            source_features,
            source_labels,
            source_domains,
            name="vrex",
        )
        self.classes_ = classes
        self.source_domains_ = domain_names
        n_classes = int(classes.shape[0])
        n_domains = int(domain_names.shape[0])

        seed = None if self.random_state is None else _integer(self.random_state, "vrex_random_state")
        hidden_units = _positive_int(self.hidden_units, "vrex_hidden_units")
        embedding_dim = _positive_int(self.embedding_dim, "vrex_embedding_dim")
        max_epochs = _positive_int(self.max_epochs, "vrex_max_epochs")
        batch_size = _positive_int(self.batch_size, "vrex_batch_size")
        patience = _positive_int(self.patience, "vrex_patience")
        learning_rate = _positive_float(self.learning_rate, "vrex_learning_rate")
        weight_decay = _nonnegative_float(self.weight_decay, "vrex_weight_decay")
        penalty_weight = _nonnegative_float(self.penalty_weight, "vrex_penalty_weight")
        anneal_epochs = _integer(self.penalty_anneal_epochs, "vrex_penalty_anneal_epochs")
        if anneal_epochs < 0:
            raise ValueError("vrex_penalty_anneal_epochs must be non-negative.")
        validation_fraction = _bounded_float(self.validation_fraction, "vrex_validation_fraction", lower=0.0, upper=1.0)
        dropout = _bounded_float(self.dropout, "vrex_dropout", lower=0.0, upper=1.0)

        torch = _torch()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        train_idx, validation_idx, validation_mode = _source_domain_validation_split(
            y,
            domains,
            validation_fraction=validation_fraction,
            random_state=seed,
        )
        device = self._resolve_device()
        model = _SourceDGModule(
            input_dim=x.shape[1],
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            n_domains=n_domains,
            dropout=dropout,
            use_domain_head=False,
        ).to(device)
        class_weights = _balanced_weights(y[train_idx], n_classes, device=device) if self.class_weight == "balanced" else None
        loss_each = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        feature_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        label_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
        domain_tensor = torch.as_tensor(domains, dtype=torch.long, device=device)

        best_loss = np.inf
        best_state = None
        best_mean_risk = np.nan
        best_risk_variance = np.nan
        patience_left = patience
        epochs_run = 0
        steps_per_epoch = max(1, int(np.ceil(train_idx.shape[0] / batch_size)))

        for epoch in range(max_epochs):
            epochs_run = epoch + 1
            model.train()
            epoch_means: list[float] = []
            epoch_variances: list[float] = []
            for _ in range(steps_per_epoch):
                batch_idx = _domain_balanced_batch(train_idx, domains, batch_size=batch_size, rng=rng)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(feature_tensor[batch_idx], grl_scale=0.0)
                sample_losses = loss_each(logits, label_tensor[batch_idx])
                batch_domains = domain_tensor[batch_idx]
                risks = torch.stack([sample_losses[batch_domains == domain].mean() for domain in torch.unique(batch_domains)])
                mean_risk = risks.mean()
                risk_variance = torch.var(risks, unbiased=False) if risks.numel() > 1 else risks.new_zeros(())
                active_penalty = penalty_weight if epoch >= anneal_epochs else 0.0
                loss = mean_risk + active_penalty * risk_variance
                loss.backward()
                optimizer.step()
                epoch_means.append(float(mean_risk.detach().cpu()))
                epoch_variances.append(float(risk_variance.detach().cpu()))

            model.eval()
            with torch.no_grad():
                valid_logits, _ = model(feature_tensor[validation_idx], grl_scale=0.0)
                valid_losses = loss_each(valid_logits, label_tensor[validation_idx])
                valid_domains = domain_tensor[validation_idx]
                domain_valid_losses = torch.stack([valid_losses[valid_domains == domain].mean() for domain in torch.unique(valid_domains)])
                validation_loss = float(domain_valid_losses.max().cpu()) if validation_mode == "heldout_source_domain" else float(valid_losses.mean().cpu())
            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_mean_risk = float(np.mean(epoch_means))
                best_risk_variance = float(np.mean(epoch_variances))
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.device_ = device
        self.n_features_in_ = int(x.shape[1])
        self.n_classes_ = n_classes
        self.n_source_domains_ = n_domains
        self.n_epochs_ = epochs_run
        self.best_source_validation_loss_ = float(best_loss)
        self.best_mean_domain_risk_ = float(best_mean_risk)
        self.best_domain_risk_variance_ = float(best_risk_variance)
        self.source_rows_ = int(x.shape[0])
        self.source_train_rows_ = int(train_idx.shape[0])
        self.source_validation_rows_ = int(validation_idx.shape[0])
        self.source_validation_mode_ = validation_mode
        return self

    def _logits(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchVRExClassifier must be fitted before prediction.")
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("features must be two-dimensional with the fitted feature width.")
        torch = _torch()
        with torch.no_grad():
            logits, _ = self.model_(torch.as_tensor(matrix, dtype=torch.float32, device=self.device_), grl_scale=0.0)
        return logits.detach().cpu().numpy()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        torch = _torch()
        return torch.softmax(torch.as_tensor(self._logits(features)), dim=1).numpy().astype(float, copy=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        logits = self._logits(features)
        return logits[:, 1] - logits[:, 0] if logits.shape[1] == 2 else logits

    def metadata(self, *, test_rows: int | None = None) -> dict[str, Any]:
        return {
            "source_vrex": True,
            "source_vrex_protocol": SOURCE_VREX_PROTOCOL,
            "source_vrex_protocol_category": SOURCE_VREX_CATEGORY,
            "source_vrex_uses_source_features": True,
            "source_vrex_uses_source_labels": True,
            "source_vrex_uses_source_domains": True,
            "source_vrex_uses_target_features": False,
            "source_vrex_uses_target_labels": False,
            "source_vrex_valid_for_strict_source_only": True,
            "source_vrex_source_rows": int(getattr(self, "source_rows_", 0)),
            "source_vrex_source_domains": int(getattr(self, "n_source_domains_", 0)),
            "source_vrex_test_rows": "" if test_rows is None else int(test_rows),
            "source_vrex_epochs_run": int(getattr(self, "n_epochs_", 0)),
            "source_vrex_validation_mode": getattr(self, "source_validation_mode_", ""),
            "source_vrex_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "source_vrex_mean_domain_risk": float(getattr(self, "best_mean_domain_risk_", np.nan)),
            "source_vrex_domain_risk_variance": float(getattr(self, "best_domain_risk_variance_", np.nan)),
            "source_vrex_penalty_weight": float(self.penalty_weight),
            "source_vrex_penalty_anneal_epochs": int(self.penalty_anneal_epochs),
            "source_vrex_device": str(getattr(self, "device_", self.device)),
        }


def fit_source_vrex_predict_proba(*, source_features: np.ndarray, source_labels: np.ndarray, source_domains: np.ndarray, test_features: np.ndarray, **kwargs: Any) -> SourceVRExFitResult:
    """Fit VREx on source domains and return probabilities for evaluation rows."""

    model = TorchVRExClassifier(**kwargs)
    model.fit(source_features, source_labels, source_domains=source_domains)
    probabilities = model.predict_proba(test_features)
    return SourceVRExFitResult(model=model, probabilities=probabilities, metadata=model.metadata(test_rows=len(test_features)))


def _domain_balanced_batch(train_idx: np.ndarray, domains: np.ndarray, *, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    domain_values = np.unique(domains[train_idx])
    per_domain = max(1, int(np.ceil(batch_size / domain_values.shape[0])))
    chunks = []
    for domain in domain_values:
        candidates = train_idx[domains[train_idx] == domain]
        chunks.append(rng.choice(candidates, size=min(per_domain, batch_size), replace=candidates.shape[0] < per_domain))
    batch = np.concatenate(chunks)
    rng.shuffle(batch)
    return batch[:batch_size]
