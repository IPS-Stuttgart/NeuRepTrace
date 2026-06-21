from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from neureptrace.decoding.dann import (
    _GradientReverse,
    _bounded_float,
    _integer,
    _nonnegative_float,
    _positive_float,
    _positive_int,
    _torch,
)


SOURCE_ADVERSARIAL_PROTOCOL = "source_only_subject_adversarial"


@dataclass(frozen=True, slots=True)
class SourceDomainGeneralizationFitResult:
    model: "TorchSourceAdversarialClassifier"
    probabilities: np.ndarray
    metadata: dict[str, Any]


class TorchSourceAdversarialClassifier(ClassifierMixin, BaseEstimator):
    """Source-only subject-adversarial neural decoder for protocol-1 domain generalization.

    The estimator uses source labels and source-domain identifiers, usually source
    subject IDs, to learn a representation that is predictive of class labels but
    hard to decode back to the source subject. It deliberately has no target-data
    argument in ``fit`` so a held-out LOSO target cannot enter representation
    learning or hyperparameter selection through this API.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        embedding_dim: int = 32,
        max_epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        domain_loss_weight: float = 0.1,
        validation_fraction: float = 0.1,
        patience: int = 10,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
        domain_weight: str | None = "balanced",
        device: str = "auto",
    ):
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.domain_loss_weight = domain_loss_weight
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.class_weight = class_weight
        self.domain_weight = domain_weight
        self.device = device

    def _resolve_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def fit(
        self,
        source_features: np.ndarray,
        source_labels: np.ndarray,
        *,
        source_domains: np.ndarray,
    ):
        x_source = np.asarray(source_features, dtype=np.float32)
        if x_source.ndim != 2:
            raise ValueError("source_adversarial source_features must be two-dimensional.")
        if x_source.shape[0] < 2:
            raise ValueError("source_adversarial needs at least two source rows.")

        y_raw = np.asarray(source_labels)
        if y_raw.shape[0] != x_source.shape[0]:
            raise ValueError("source_features and source_labels must contain the same rows.")
        self.classes_, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64, copy=False)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError("source_adversarial needs at least two source classes.")

        domain_raw = np.asarray(source_domains, dtype=object).reshape(-1)
        if domain_raw.shape[0] != x_source.shape[0]:
            raise ValueError("source_features and source_domains must contain the same rows.")
        if np.any(_is_missing_domain_array(domain_raw)):
            raise ValueError("source_domains must not contain missing values.")
        self.source_domains_, domains = np.unique(domain_raw.astype(str), return_inverse=True)
        domains = domains.astype(np.int64, copy=False)
        n_domains = int(self.source_domains_.shape[0])
        if n_domains < 2:
            raise ValueError("source_adversarial needs at least two source domains/subjects.")

        random_state = None if self.random_state is None else _integer(
            self.random_state,
            "source_adversarial_random_state",
        )
        torch = _torch()
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

        hidden_units = _positive_int(self.hidden_units, "source_adversarial_hidden_units")
        embedding_dim = _positive_int(self.embedding_dim, "source_adversarial_embedding_dim")
        max_epochs = _positive_int(self.max_epochs, "source_adversarial_max_epochs")
        batch_size = _positive_int(self.batch_size, "source_adversarial_batch_size")
        patience = _positive_int(self.patience, "source_adversarial_patience")
        learning_rate = _positive_float(self.learning_rate, "source_adversarial_learning_rate")
        weight_decay = _nonnegative_float(self.weight_decay, "source_adversarial_weight_decay")
        domain_loss_weight = _nonnegative_float(
            self.domain_loss_weight,
            "source_adversarial_domain_loss_weight",
        )
        dropout = _bounded_float(self.dropout, "source_adversarial_dropout", lower=0.0, upper=1.0)
        validation_fraction = _bounded_float(
            self.validation_fraction,
            "source_adversarial_validation_fraction",
            lower=0.0,
            upper=1.0,
        )

        source_indices = np.arange(y.shape[0])
        class_counts = np.bincount(y, minlength=n_classes)
        can_validate = (
            0.0 < validation_fraction < 1.0
            and y.shape[0] >= 2 * n_classes
            and np.min(class_counts) >= 2
        )
        if can_validate:
            train_idx, validation_idx = train_test_split(
                source_indices,
                test_size=validation_fraction,
                random_state=random_state,
                stratify=y,
            )
        else:
            train_idx = source_indices
            validation_idx = source_indices

        device = self._resolve_device()
        model = _SourceAdversarialModule(
            input_dim=x_source.shape[1],
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            n_domains=n_domains,
            dropout=dropout,
        ).to(device)

        class_loss_fn = torch.nn.CrossEntropyLoss(
            weight=_balanced_weights(y[train_idx], n_classes, device=device)
            if self.class_weight == "balanced"
            else None
        )
        domain_loss_fn = torch.nn.CrossEntropyLoss(
            weight=_balanced_weights(domains[train_idx], n_domains, device=device)
            if self.domain_weight == "balanced"
            else None
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        source_tensor = torch.as_tensor(x_source, dtype=torch.float32, device=device)
        source_label_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
        source_domain_tensor = torch.as_tensor(domains, dtype=torch.long, device=device)
        rng = np.random.default_rng(random_state)
        best_loss = np.inf
        best_state = None
        patience_left = patience
        epochs_run = 0
        steps_per_epoch = max(1, int(np.ceil(train_idx.shape[0] / batch_size)))

        for epoch in range(max_epochs):
            epochs_run = epoch + 1
            model.train()
            for _step in range(steps_per_epoch):
                source_batch = rng.choice(
                    train_idx,
                    size=min(batch_size, train_idx.shape[0]),
                    replace=train_idx.shape[0] < batch_size,
                )
                batch_source = source_tensor[source_batch]
                batch_source_labels = source_label_tensor[source_batch]
                batch_source_domains = source_domain_tensor[source_batch]

                optimizer.zero_grad(set_to_none=True)
                source_class_logits, source_domain_logits = model(batch_source, grl_scale=1.0)
                class_loss = class_loss_fn(source_class_logits, batch_source_labels)
                domain_loss = domain_loss_fn(source_domain_logits, batch_source_domains)
                loss = class_loss + domain_loss_weight * domain_loss
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_logits, _ = model(source_tensor[validation_idx], grl_scale=0.0)
                validation_loss = float(class_loss_fn(validation_logits, source_label_tensor[validation_idx]).detach().cpu())
            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.device_ = device
        self.n_features_in_ = x_source.shape[1]
        self.n_classes_ = n_classes
        self.n_source_domains_ = n_domains
        self.n_epochs_ = epochs_run
        self.best_source_validation_loss_ = float(best_loss)
        self.source_rows_ = int(x_source.shape[0])
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        logits = self._logits(features)
        if logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        return logits

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        torch = _torch()
        logits = torch.as_tensor(self._logits(features), dtype=torch.float32)
        return torch.softmax(logits, dim=1).detach().cpu().numpy().astype(float, copy=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def _logits(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchSourceAdversarialClassifier must be fitted before prediction.")
        torch = _torch()
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(x, grl_scale=0.0)
        return logits.detach().cpu().numpy()

    def metadata(self, *, test_rows: int | None = None) -> dict[str, Any]:
        return {
            "source_adversarial_domain_generalization": True,
            "source_adversarial_protocol": SOURCE_ADVERSARIAL_PROTOCOL,
            "source_adversarial_uses_target_features": False,
            "source_adversarial_uses_target_labels": False,
            "source_adversarial_valid_for_benchmark": True,
            "source_adversarial_hidden_units": int(self.hidden_units),
            "source_adversarial_embedding_dim": int(self.embedding_dim),
            "source_adversarial_max_epochs": int(self.max_epochs),
            "source_adversarial_epochs_run": int(getattr(self, "n_epochs_", 0)),
            "source_adversarial_batch_size": int(self.batch_size),
            "source_adversarial_learning_rate": float(self.learning_rate),
            "source_adversarial_weight_decay": float(self.weight_decay),
            "source_adversarial_domain_loss_weight": float(self.domain_loss_weight),
            "source_adversarial_patience": int(self.patience),
            "source_adversarial_dropout": float(self.dropout),
            "source_adversarial_source_rows": int(getattr(self, "source_rows_", 0)),
            "source_adversarial_test_rows": "" if test_rows is None else int(test_rows),
            "source_adversarial_source_domains": int(getattr(self, "n_source_domains_", 0)),
            "source_adversarial_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "source_adversarial_device": str(getattr(self, "device_", self.device)),
        }


class _SourceAdversarialModule:
    def __new__(
        cls,
        *,
        input_dim: int,
        hidden_units: int,
        embedding_dim: int,
        n_classes: int,
        n_domains: int,
        dropout: float,
    ):
        torch = _torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_units),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_units, embedding_dim),
                    torch.nn.ReLU(),
                )
                self.class_head = torch.nn.Linear(embedding_dim, n_classes)
                self.domain_head = torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, hidden_units),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_units, n_domains),
                )

            def forward(self, features, *, grl_scale: float):
                embedding = self.feature_extractor(features)
                class_logits = self.class_head(embedding)
                reversed_embedding = _GradientReverse.apply(embedding, float(grl_scale))
                domain_logits = self.domain_head(reversed_embedding)
                return class_logits, domain_logits

        return Module()


def fit_source_adversarial_predict_proba(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_domains: np.ndarray,
    test_features: np.ndarray,
    hidden_units: int = 64,
    embedding_dim: int = 32,
    max_epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    domain_loss_weight: float = 0.1,
    validation_fraction: float = 0.1,
    patience: int = 10,
    dropout: float = 0.1,
    random_state: int | None = 13,
    device: str = "auto",
) -> SourceDomainGeneralizationFitResult:
    source_features = np.asarray(source_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    if source_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("source_adversarial source_features and test_features must be two-dimensional.")
    if source_features.shape[1] != test_features.shape[1]:
        raise ValueError("source_adversarial train and test features must have the same feature dimension.")

    model = TorchSourceAdversarialClassifier(
        hidden_units=hidden_units,
        embedding_dim=embedding_dim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        domain_loss_weight=domain_loss_weight,
        validation_fraction=validation_fraction,
        patience=patience,
        dropout=dropout,
        random_state=random_state,
        device=device,
    )
    model.fit(source_features, source_labels, source_domains=source_domains)
    probabilities = model.predict_proba(test_features)
    return SourceDomainGeneralizationFitResult(
        model=model,
        probabilities=probabilities,
        metadata=model.metadata(test_rows=test_features.shape[0]),
    )


def _balanced_weights(encoded_labels: np.ndarray, n_levels: int, *, device):
    torch = _torch()
    counts = np.bincount(np.asarray(encoded_labels, dtype=int), minlength=n_levels).astype(np.float32)
    weights = len(encoded_labels) / np.maximum(counts, 1.0) / float(n_levels)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _is_missing_domain_array(values: np.ndarray) -> np.ndarray:
    """Small pandas-free missing-value check for object/string domain IDs."""

    flattened = np.asarray(values, dtype=object).reshape(-1)
    return np.asarray(
        [value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in flattened],
        dtype=bool,
    )
