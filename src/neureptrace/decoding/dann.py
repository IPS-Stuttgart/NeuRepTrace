from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


DANN_PROTOCOL = "unlabeled_target_domain_adversarial"


@dataclass(frozen=True, slots=True)
class DANNFitResult:
    model: "TorchDANNClassifier"
    probabilities: np.ndarray
    metadata: dict[str, Any]


class _GradientReverse:
    @staticmethod
    def apply(features, scale: float):
        torch = _torch()

        class _GradientReverseFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                ctx.scale = float(scale)
                return input_tensor.view_as(input_tensor)

            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.scale * grad_output, None

        return _GradientReverseFunction.apply(features)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("The 'dann' decoder requires torch, e.g. `pip install neureptrace[torch]`.") from exc
    return torch


class TorchDANNClassifier(ClassifierMixin, BaseEstimator):
    """Small domain-adversarial neural network for category-2 adaptation.

    ``fit`` uses source labels and unlabeled target features. Target labels are
    intentionally not accepted by this estimator.
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
        target_features: np.ndarray,
    ):
        x_source = np.asarray(source_features, dtype=np.float32)
        x_target = np.asarray(target_features, dtype=np.float32)
        if x_source.ndim != 2 or x_target.ndim != 2:
            raise ValueError("DANN source_features and target_features must be two-dimensional.")
        if x_source.shape[1] != x_target.shape[1]:
            raise ValueError("DANN source and target features must have the same feature dimension.")
        if x_source.shape[0] < 2 or x_target.shape[0] < 1:
            raise ValueError("DANN needs at least two source rows and one target row.")

        y_raw = np.asarray(source_labels)
        if y_raw.shape[0] != x_source.shape[0]:
            raise ValueError("DANN source_features and source_labels must contain the same rows.")
        self.classes_, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64, copy=False)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError("DANN needs at least two source classes.")

        torch = _torch()
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

        hidden_units = _positive_int(self.hidden_units, "dann_hidden_units")
        embedding_dim = _positive_int(self.embedding_dim, "dann_embedding_dim")
        max_epochs = _positive_int(self.max_epochs, "dann_max_epochs")
        batch_size = _positive_int(self.batch_size, "dann_batch_size")
        patience = _positive_int(self.patience, "dann_patience")
        learning_rate = _positive_float(self.learning_rate, "dann_learning_rate")
        weight_decay = _nonnegative_float(self.weight_decay, "dann_weight_decay")
        domain_loss_weight = _nonnegative_float(self.domain_loss_weight, "dann_domain_loss_weight")
        dropout = _bounded_float(self.dropout, "dann_dropout", lower=0.0, upper=1.0)

        source_indices = np.arange(y.shape[0])
        class_counts = np.bincount(y, minlength=n_classes)
        can_validate = (
            0.0 < float(self.validation_fraction) < 1.0
            and y.shape[0] >= 2 * n_classes
            and np.min(class_counts) >= 2
        )
        if can_validate:
            train_idx, validation_idx = train_test_split(
                source_indices,
                test_size=float(self.validation_fraction),
                random_state=self.random_state,
                stratify=y,
            )
        else:
            train_idx = source_indices
            validation_idx = source_indices

        device = self._resolve_device()
        model = _DANNModule(
            input_dim=x_source.shape[1],
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            dropout=dropout,
        ).to(device)

        if self.class_weight == "balanced":
            train_counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
            weights = train_idx.shape[0] / np.maximum(train_counts, 1.0) / float(n_classes)
            class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        else:
            class_weights = None
        class_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        domain_loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        source_tensor = torch.as_tensor(x_source, dtype=torch.float32, device=device)
        source_label_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
        target_tensor = torch.as_tensor(x_target, dtype=torch.float32, device=device)
        rng = np.random.default_rng(self.random_state)
        best_loss = np.inf
        best_state = None
        patience_left = patience
        epochs_run = 0
        steps_per_epoch = max(
            1,
            int(np.ceil(train_idx.shape[0] / batch_size)),
            int(np.ceil(x_target.shape[0] / batch_size)),
        )

        for epoch in range(max_epochs):
            epochs_run = epoch + 1
            model.train()
            for _step in range(steps_per_epoch):
                source_batch = rng.choice(train_idx, size=min(batch_size, train_idx.shape[0]), replace=train_idx.shape[0] < batch_size)
                target_batch = rng.choice(x_target.shape[0], size=min(batch_size, x_target.shape[0]), replace=x_target.shape[0] < batch_size)
                batch_source = source_tensor[source_batch]
                batch_source_labels = source_label_tensor[source_batch]
                batch_target = target_tensor[target_batch]

                optimizer.zero_grad(set_to_none=True)
                source_class_logits, source_domain_logits = model(
                    batch_source,
                    grl_scale=1.0,
                )
                _, target_domain_logits = model(batch_target, grl_scale=1.0)
                class_loss = class_loss_fn(source_class_logits, batch_source_labels)
                source_domain_labels = torch.zeros(source_domain_logits.shape[0], dtype=torch.long, device=device)
                target_domain_labels = torch.ones(target_domain_logits.shape[0], dtype=torch.long, device=device)
                domain_loss = 0.5 * (
                    domain_loss_fn(source_domain_logits, source_domain_labels)
                    + domain_loss_fn(target_domain_logits, target_domain_labels)
                )
                loss = class_loss + domain_loss_weight * domain_loss
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_logits, _ = model(source_tensor[validation_idx], grl_scale=0.0)
                validation_loss = float(
                    class_loss_fn(validation_logits, source_label_tensor[validation_idx]).detach().cpu()
                )
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
        self.n_epochs_ = epochs_run
        self.best_source_validation_loss_ = float(best_loss)
        self.source_rows_ = int(x_source.shape[0])
        self.target_rows_ = int(x_target.shape[0])
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
            raise RuntimeError("TorchDANNClassifier must be fitted before prediction.")
        torch = _torch()
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(x, grl_scale=0.0)
        return logits.detach().cpu().numpy()

    def metadata(self) -> dict[str, Any]:
        return {
            "dann_adaptation": True,
            "dann_protocol": DANN_PROTOCOL,
            "dann_uses_target_features": True,
            "dann_uses_target_labels": False,
            "dann_valid_for_benchmark": True,
            "dann_hidden_units": int(self.hidden_units),
            "dann_embedding_dim": int(self.embedding_dim),
            "dann_max_epochs": int(self.max_epochs),
            "dann_epochs_run": int(getattr(self, "n_epochs_", 0)),
            "dann_batch_size": int(self.batch_size),
            "dann_learning_rate": float(self.learning_rate),
            "dann_weight_decay": float(self.weight_decay),
            "dann_domain_loss_weight": float(self.domain_loss_weight),
            "dann_patience": int(self.patience),
            "dann_dropout": float(self.dropout),
            "dann_source_rows": int(getattr(self, "source_rows_", 0)),
            "dann_target_rows": int(getattr(self, "target_rows_", 0)),
            "dann_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "dann_device": str(getattr(self, "device_", self.device)),
        }


class _DANNModule:
    def __new__(
        cls,
        *,
        input_dim: int,
        hidden_units: int,
        embedding_dim: int,
        n_classes: int,
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
                    torch.nn.Linear(hidden_units, 2),
                )

            def forward(self, features, *, grl_scale: float):
                embedding = self.feature_extractor(features)
                class_logits = self.class_head(embedding)
                reversed_embedding = _GradientReverse.apply(embedding, float(grl_scale))
                domain_logits = self.domain_head(reversed_embedding)
                return class_logits, domain_logits

        return Module()


def fit_dann_predict_proba(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
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
) -> DANNFitResult:
    model = TorchDANNClassifier(
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
    model.fit(source_features, source_labels, target_features=target_features)
    probabilities = model.predict_proba(target_features)
    return DANNFitResult(model=model, probabilities=probabilities, metadata=model.metadata())


def _positive_int(value: Any, name: str) -> int:
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return number


def _bounded_float(value: Any, name: str, *, lower: float, upper: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < lower or number >= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}).")
    return number
