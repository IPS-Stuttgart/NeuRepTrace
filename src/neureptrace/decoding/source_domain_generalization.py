from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
SOURCE_GROUP_DRO_PROTOCOL = "source_only_group_dro"
SOURCE_ERM_PROTOCOL = "source_only_neural_erm"
SOURCE_DG_STRATEGIES = ("subject_adversarial", "group_dro", "erm")
SourceDGStrategy = Literal["subject_adversarial", "group_dro", "erm"]


@dataclass(frozen=True, slots=True)
class SourceDomainGeneralizationFitResult:
    model: ClassifierMixin
    probabilities: np.ndarray
    metadata: dict[str, Any]


def normalize_source_domain_generalization_strategy(strategy: str | None) -> SourceDGStrategy:
    normalized = "subject_adversarial" if strategy is None else str(strategy).strip().lower().replace("-", "_")
    aliases = {
        "source_adversarial": "subject_adversarial",
        "adversarial": "subject_adversarial",
        "subject_adversarial": "subject_adversarial",
        "domain_adversarial": "subject_adversarial",
        "groupdro": "group_dro",
        "group_dro": "group_dro",
        "dro": "group_dro",
        "erm": "erm",
        "neural_erm": "erm",
        "source_erm": "erm",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unknown source-domain generalization strategy "
            f"'{strategy}'. Available strategies: {', '.join(SOURCE_DG_STRATEGIES)}."
        )
    return aliases[normalized]  # type: ignore[return-value]


def _protocol_for_strategy(strategy: str) -> str:
    if strategy == "subject_adversarial":
        return SOURCE_ADVERSARIAL_PROTOCOL
    if strategy == "group_dro":
        return SOURCE_GROUP_DRO_PROTOCOL
    return SOURCE_ERM_PROTOCOL


def _encode_inputs(source_features: np.ndarray, source_labels: np.ndarray, source_domains: np.ndarray, *, name: str):
    x = np.asarray(source_features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} source_features must be two-dimensional.")
    if x.shape[0] < 2:
        raise ValueError(f"{name} needs at least two source rows.")

    y_raw = np.asarray(source_labels)
    if y_raw.shape[0] != x.shape[0]:
        raise ValueError("source_features and source_labels must contain the same rows.")
    classes, y = np.unique(y_raw, return_inverse=True)
    y = y.astype(np.int64, copy=False)
    if classes.shape[0] < 2:
        raise ValueError(f"{name} needs at least two source classes.")

    domain_raw = np.asarray(source_domains, dtype=object).reshape(-1)
    if domain_raw.shape[0] != x.shape[0]:
        raise ValueError("source_features and source_domains must contain the same rows.")
    if np.any(_is_missing_domain_array(domain_raw)):
        raise ValueError("source_domains must not contain missing values.")
    domain_names, domains = np.unique(domain_raw.astype(str), return_inverse=True)
    domains = domains.astype(np.int64, copy=False)
    if domain_names.shape[0] < 2:
        raise ValueError(f"{name} needs at least two source domains/subjects.")
    return x, classes, y, domain_names, domains


def _source_domain_validation_split(labels: np.ndarray, domains: np.ndarray, *, validation_fraction: float, random_state: int | None):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    domains = np.asarray(domains, dtype=np.int64).reshape(-1)
    indices = np.arange(labels.shape[0])
    unique_domains = np.unique(domains)
    n_classes = int(np.unique(labels).shape[0])
    fraction = float(validation_fraction)
    rng = np.random.default_rng(random_state)

    if 0.0 < fraction < 1.0 and unique_domains.shape[0] >= 2:
        n_valid_domains = max(1, int(round(unique_domains.shape[0] * fraction)))
        n_valid_domains = min(n_valid_domains, unique_domains.shape[0] - 1)
        subsets = [(domain,) for domain in rng.permutation(unique_domains).tolist()]
        if n_valid_domains > 1:
            for _ in range(min(32, 4 * unique_domains.shape[0])):
                subset = tuple(sorted(rng.choice(unique_domains, size=n_valid_domains, replace=False).tolist()))
                if subset not in subsets:
                    subsets.append(subset)
        for valid_domains in subsets:
            valid_mask = np.isin(domains, valid_domains)
            train_idx = indices[~valid_mask]
            valid_idx = indices[valid_mask]
            if train_idx.size and valid_idx.size:
                train_has_all_classes = np.unique(labels[train_idx]).shape[0] == n_classes
                valid_has_two_classes = np.unique(labels[valid_idx]).shape[0] >= 2
                if train_has_all_classes and valid_has_two_classes:
                    return train_idx, valid_idx, "heldout_source_domain"

    class_counts = np.bincount(labels, minlength=n_classes)
    can_row_validate = (
        0.0 < fraction < 1.0
        and labels.shape[0] >= 2 * n_classes
        and np.min(class_counts) >= 2
    )
    if can_row_validate:
        train_idx, valid_idx = train_test_split(indices, test_size=fraction, random_state=random_state, stratify=labels)
        return train_idx, valid_idx, "stratified_row_fallback"
    return indices, indices, "training_loss_fallback"


class TorchSourceDomainGeneralizationClassifier(ClassifierMixin, BaseEstimator):
    """Source-only neural domain generalization decoder for strict LOSO.

    ``strategy='subject_adversarial'`` uses a gradient-reversal source-subject
    head, ``strategy='group_dro'`` uses robust source-domain reweighting, and
    ``strategy='erm'`` is a matched neural ERM baseline.  All strategies use
    only source features, source labels, and source domain IDs; no target
    features or target labels are accepted by ``fit``.
    """

    def __init__(
        self,
        strategy: str = "subject_adversarial",
        hidden_units: int = 64,
        embedding_dim: int = 32,
        max_epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        domain_loss_weight: float = 0.1,
        group_dro_eta: float = 0.05,
        validation_fraction: float = 0.1,
        patience: int = 10,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
        domain_weight: str | None = "balanced",
        device: str = "auto",
    ):
        self.strategy = strategy
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.domain_loss_weight = domain_loss_weight
        self.group_dro_eta = group_dro_eta
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

    def fit(self, source_features: np.ndarray, source_labels: np.ndarray, *, source_domains: np.ndarray):
        strategy = normalize_source_domain_generalization_strategy(self.strategy)
        x, classes, y, domain_names, domains = _encode_inputs(source_features, source_labels, source_domains, name=strategy)
        self.classes_ = classes
        self.source_domains_ = domain_names
        n_classes = int(classes.shape[0])
        n_domains = int(domain_names.shape[0])

        random_state = None if self.random_state is None else _integer(self.random_state, "source_dg_random_state")
        torch = _torch()
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

        hidden_units = _positive_int(self.hidden_units, "source_dg_hidden_units")
        embedding_dim = _positive_int(self.embedding_dim, "source_dg_embedding_dim")
        max_epochs = _positive_int(self.max_epochs, "source_dg_max_epochs")
        batch_size = _positive_int(self.batch_size, "source_dg_batch_size")
        patience = _positive_int(self.patience, "source_dg_patience")
        learning_rate = _positive_float(self.learning_rate, "source_dg_learning_rate")
        weight_decay = _nonnegative_float(self.weight_decay, "source_dg_weight_decay")
        domain_loss_weight = _nonnegative_float(self.domain_loss_weight, "source_dg_domain_loss_weight")
        group_dro_eta = _positive_float(self.group_dro_eta, "source_dg_group_dro_eta")
        dropout = _bounded_float(self.dropout, "source_dg_dropout", lower=0.0, upper=1.0)
        validation_fraction = _bounded_float(self.validation_fraction, "source_dg_validation_fraction", lower=0.0, upper=1.0)
        train_idx, validation_idx, validation_mode = _source_domain_validation_split(
            y,
            domains,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )

        device = self._resolve_device()
        model = _SourceDGModule(
            input_dim=x.shape[1],
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            n_domains=n_domains,
            dropout=dropout,
            use_domain_head=strategy == "subject_adversarial",
        ).to(device)
        class_weights = _balanced_weights(y[train_idx], n_classes, device=device) if self.class_weight == "balanced" else None
        class_loss_mean = torch.nn.CrossEntropyLoss(weight=class_weights)
        class_loss_each = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        domain_loss_fn = torch.nn.CrossEntropyLoss(
            weight=_balanced_weights(domains[train_idx], n_domains, device=device) if self.domain_weight == "balanced" else None
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        source_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        label_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
        domain_tensor = torch.as_tensor(domains, dtype=torch.long, device=device)
        group_weights = torch.full((n_domains,), 1.0 / float(n_domains), dtype=torch.float32, device=device)
        rng = np.random.default_rng(random_state)
        best_loss = np.inf
        best_state = None
        best_group_weights = None
        patience_left = patience
        epochs_run = 0
        steps_per_epoch = max(1, int(np.ceil(train_idx.shape[0] / batch_size)))

        for epoch in range(max_epochs):
            epochs_run = epoch + 1
            model.train()
            for _ in range(steps_per_epoch):
                batch_idx = rng.choice(train_idx, size=min(batch_size, train_idx.shape[0]), replace=train_idx.shape[0] < batch_size)
                optimizer.zero_grad(set_to_none=True)
                class_logits, domain_logits = model(source_tensor[batch_idx], grl_scale=1.0)
                if strategy == "group_dro":
                    per_sample_loss = class_loss_each(class_logits, label_tensor[batch_idx])
                    batch_domains = domain_tensor[batch_idx]
                    present_domains = torch.unique(batch_domains)
                    per_group_loss = torch.stack([per_sample_loss[batch_domains == d].mean() for d in present_domains])
                    present_weights = group_weights[present_domains]
                    loss = torch.sum((present_weights / present_weights.sum()) * per_group_loss)
                    with torch.no_grad():
                        group_weights[present_domains] *= torch.exp(group_dro_eta * per_group_loss.detach())
                        group_weights /= group_weights.sum().clamp_min(1e-12)
                else:
                    loss = class_loss_mean(class_logits, label_tensor[batch_idx])
                    if strategy == "subject_adversarial":
                        loss = loss + domain_loss_weight * domain_loss_fn(domain_logits, domain_tensor[batch_idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_logits, _ = model(source_tensor[validation_idx], grl_scale=0.0)
                valid_losses = class_loss_each(valid_logits, label_tensor[validation_idx])
                if validation_mode == "heldout_source_domain":
                    valid_domains = domain_tensor[validation_idx]
                    validation_loss = float(torch.stack([valid_losses[valid_domains == d].mean() for d in torch.unique(valid_domains)]).max().cpu())
                else:
                    validation_loss = float(valid_losses.mean().cpu())
            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_group_weights = group_weights.detach().cpu().numpy().astype(float, copy=True)
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.device_ = device
        self.strategy_ = strategy
        self.n_features_in_ = x.shape[1]
        self.n_classes_ = n_classes
        self.n_source_domains_ = n_domains
        self.n_epochs_ = epochs_run
        self.best_source_validation_loss_ = float(best_loss)
        self.source_rows_ = int(x.shape[0])
        self.source_train_rows_ = int(train_idx.shape[0])
        self.source_validation_rows_ = int(validation_idx.shape[0])
        self.source_validation_mode_ = validation_mode
        self.group_weights_ = best_group_weights if best_group_weights is not None else group_weights.detach().cpu().numpy().astype(float)
        return self

    def _logits(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchSourceDomainGeneralizationClassifier must be fitted before prediction.")
        torch = _torch()
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(x, grl_scale=0.0)
        return logits.detach().cpu().numpy()

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

    def metadata(self, *, test_rows: int | None = None) -> dict[str, Any]:
        strategy = getattr(self, "strategy_", normalize_source_domain_generalization_strategy(self.strategy))
        protocol = _protocol_for_strategy(strategy)
        group_weights = np.asarray(getattr(self, "group_weights_", np.array([], dtype=float))).ravel()
        metadata = {
            "source_domain_generalization": True,
            "source_domain_generalization_strategy": strategy,
            "source_domain_generalization_protocol": protocol,
            "source_domain_generalization_uses_target_features": False,
            "source_domain_generalization_uses_target_labels": False,
            "source_domain_generalization_valid_for_benchmark": True,
            "source_domain_generalization_validation_mode": getattr(self, "source_validation_mode_", ""),
            "source_domain_generalization_source_rows": int(getattr(self, "source_rows_", 0)),
            "source_domain_generalization_source_train_rows": int(getattr(self, "source_train_rows_", 0)),
            "source_domain_generalization_source_validation_rows": int(getattr(self, "source_validation_rows_", 0)),
            "source_domain_generalization_source_domains": int(getattr(self, "n_source_domains_", 0)),
            "source_domain_generalization_test_rows": "" if test_rows is None else int(test_rows),
            "source_domain_generalization_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "source_domain_generalization_hidden_units": int(self.hidden_units),
            "source_domain_generalization_embedding_dim": int(self.embedding_dim),
            "source_domain_generalization_max_epochs": int(self.max_epochs),
            "source_domain_generalization_epochs_run": int(getattr(self, "n_epochs_", 0)),
            "source_domain_generalization_batch_size": int(self.batch_size),
            "source_domain_generalization_learning_rate": float(self.learning_rate),
            "source_domain_generalization_weight_decay": float(self.weight_decay),
            "source_domain_generalization_dropout": float(self.dropout),
            "source_domain_generalization_device": str(getattr(self, "device_", self.device)),
        }
        if strategy == "subject_adversarial":
            metadata.update(
                {
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
                    "source_adversarial_source_validation_mode": getattr(self, "source_validation_mode_", ""),
                    "source_adversarial_device": str(getattr(self, "device_", self.device)),
                }
            )
        if strategy == "group_dro":
            metadata.update(
                {
                    "group_dro_domain_generalization": True,
                    "group_dro_protocol": SOURCE_GROUP_DRO_PROTOCOL,
                    "group_dro_uses_target_features": False,
                    "group_dro_uses_target_labels": False,
                    "group_dro_valid_for_benchmark": True,
                    "group_dro_eta": float(self.group_dro_eta),
                    "group_dro_source_domains": int(getattr(self, "n_source_domains_", 0)),
                    "group_dro_final_group_weights": "|".join(f"{float(v):.6g}" for v in group_weights),
                    "group_dro_source_validation_mode": getattr(self, "source_validation_mode_", ""),
                }
            )
        return metadata


class TorchSourceAdversarialClassifier(TorchSourceDomainGeneralizationClassifier):
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
        super().__init__(
            strategy="subject_adversarial",
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
            class_weight=class_weight,
            domain_weight=domain_weight,
            device=device,
        )


class TorchGroupDROClassifier(TorchSourceDomainGeneralizationClassifier):
    def __init__(
        self,
        hidden_units: int = 64,
        embedding_dim: int = 32,
        max_epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        group_dro_eta: float = 0.05,
        validation_fraction: float = 0.1,
        patience: int = 10,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
        device: str = "auto",
    ):
        super().__init__(
            strategy="group_dro",
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            group_dro_eta=group_dro_eta,
            validation_fraction=validation_fraction,
            patience=patience,
            dropout=dropout,
            random_state=random_state,
            class_weight=class_weight,
            device=device,
        )


class TorchSourceERMClassifier(TorchSourceDomainGeneralizationClassifier):
    def __init__(
        self,
        hidden_units: int = 64,
        embedding_dim: int = 32,
        max_epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.1,
        patience: int = 10,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
        device: str = "auto",
    ):
        super().__init__(
            strategy="erm",
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_fraction=validation_fraction,
            patience=patience,
            dropout=dropout,
            random_state=random_state,
            class_weight=class_weight,
            device=device,
        )


class _SourceDGModule:
    def __new__(cls, *, input_dim: int, hidden_units: int, embedding_dim: int, n_classes: int, n_domains: int, dropout: float, use_domain_head: bool):
        torch = _torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_domain_head = use_domain_head
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
                ) if use_domain_head else None

            def forward(self, features, *, grl_scale: float):
                embedding = self.feature_extractor(features)
                class_logits = self.class_head(embedding)
                if self.domain_head is None:
                    return class_logits, None
                domain_logits = self.domain_head(_GradientReverse.apply(embedding, float(grl_scale)))
                return class_logits, domain_logits

        return Module()


def fit_source_adversarial_predict_proba(**kwargs) -> SourceDomainGeneralizationFitResult:
    kwargs = dict(kwargs)
    kwargs["strategy"] = "subject_adversarial"
    return fit_source_domain_generalization_predict_proba(**kwargs)


def fit_source_domain_generalization_predict_proba(
    *,
    strategy: str = "subject_adversarial",
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
    group_dro_eta: float = 0.05,
    validation_fraction: float = 0.1,
    patience: int = 10,
    dropout: float = 0.1,
    random_state: int | None = 13,
    device: str = "auto",
) -> SourceDomainGeneralizationFitResult:
    source_features = np.asarray(source_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    if source_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("source_domain_generalization source_features and test_features must be two-dimensional.")
    if source_features.shape[1] != test_features.shape[1]:
        raise ValueError("source_domain_generalization train and test features must have the same feature dimension.")
    model = TorchSourceDomainGeneralizationClassifier(
        strategy=strategy,
        hidden_units=hidden_units,
        embedding_dim=embedding_dim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        domain_loss_weight=domain_loss_weight,
        group_dro_eta=group_dro_eta,
        validation_fraction=validation_fraction,
        patience=patience,
        dropout=dropout,
        random_state=random_state,
        device=device,
    )
    model.fit(source_features, source_labels, source_domains=source_domains)
    probabilities = model.predict_proba(test_features)
    return SourceDomainGeneralizationFitResult(model=model, probabilities=probabilities, metadata=model.metadata(test_rows=test_features.shape[0]))


def _balanced_weights(encoded_labels: np.ndarray, n_levels: int, *, device):
    torch = _torch()
    counts = np.bincount(np.asarray(encoded_labels, dtype=int), minlength=n_levels).astype(np.float32)
    weights = len(encoded_labels) / np.maximum(counts, 1.0) / float(n_levels)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _is_missing_domain_array(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=object).reshape(-1)
    return np.asarray(
        [value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in flattened],
        dtype=bool,
    )
