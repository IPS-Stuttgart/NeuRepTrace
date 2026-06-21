from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from neureptrace.decoding.dann import (
    _GradientReverse,
    _bounded_float,
    _conditional_mmd_loss,
    _integer,
    _mmd_loss,
    _nonnegative_float,
    _positive_float,
    _positive_int,
    _torch,
)

CDAN_PROTOCOL = "unlabeled_target_conditional_domain_adversarial"
CDAN_MMD_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_mmd"
CDAN_CONDITIONAL_MMD_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_conditional_mmd"
CDAN_MMD_CONDITIONAL_MMD_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_mmd_conditional_mmd"
CDAN_DANN_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_domain_adversarial"
CDAN_DANN_MMD_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_domain_adversarial_mmd"
CDAN_DANN_CONDITIONAL_MMD_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_domain_adversarial_conditional_mmd"
CDAN_DANN_MMD_CONDITIONAL_MMD_PROTOCOL = "unlabeled_target_conditional_domain_adversarial_domain_adversarial_mmd_conditional_mmd"


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    return _positive_int(value, name)


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


@dataclass(frozen=True, slots=True)
class CDANFitResult:
    model: "TorchCDANClassifier"
    probabilities: np.ndarray
    metadata: dict[str, Any]


class TorchCDANClassifier(ClassifierMixin, BaseEstimator):
    """Conditional adversarial Category-2 decoder.

    The estimator fits a source-supervised classifier while adapting to unlabeled
    target features through a Conditional Domain Adversarial Network (CDAN)
    discriminator. The CDAN discriminator receives the multilinear map between
    learned embeddings and class-posterior probabilities. Optional marginal DANN,
    marginal MMD, and soft class-conditional MMD losses can be enabled without
    changing the target-label hygiene: ``fit`` accepts ``target_features`` but no
    target labels.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        embedding_dim: int = 32,
        max_epochs: int = 80,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        cdan_loss_weight: float = 0.1,
        domain_loss_weight: float = 0.0,
        mmd_loss_weight: float = 0.0,
        conditional_mmd_loss_weight: float = 0.0,
        cdan_randomized_dim: int | None = None,
        cdan_entropy_conditioning: bool = True,
        mmd_kernel_mul: float = 2.0,
        mmd_kernel_num: int = 5,
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
        self.cdan_loss_weight = cdan_loss_weight
        self.domain_loss_weight = domain_loss_weight
        self.mmd_loss_weight = mmd_loss_weight
        self.conditional_mmd_loss_weight = conditional_mmd_loss_weight
        self.cdan_randomized_dim = cdan_randomized_dim
        self.cdan_entropy_conditioning = cdan_entropy_conditioning
        self.mmd_kernel_mul = mmd_kernel_mul
        self.mmd_kernel_num = mmd_kernel_num
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
            raise ValueError("CDAN source_features and target_features must be two-dimensional.")
        if x_source.shape[1] != x_target.shape[1]:
            raise ValueError("CDAN source and target features must have the same feature dimension.")
        if x_source.shape[0] < 2 or x_target.shape[0] < 1:
            raise ValueError("CDAN needs at least two source rows and one target row.")

        y_raw = np.asarray(source_labels)
        if y_raw.shape[0] != x_source.shape[0]:
            raise ValueError("CDAN source_features and source_labels must contain the same rows.")
        self.classes_, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64, copy=False)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError("CDAN needs at least two source classes.")

        random_state = None if self.random_state is None else _integer(self.random_state, "cdan_random_state")
        torch = _torch()
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

        hidden_units = _positive_int(self.hidden_units, "cdan_hidden_units")
        embedding_dim = _positive_int(self.embedding_dim, "cdan_embedding_dim")
        max_epochs = _positive_int(self.max_epochs, "cdan_max_epochs")
        batch_size = _positive_int(self.batch_size, "cdan_batch_size")
        patience = _positive_int(self.patience, "cdan_patience")
        learning_rate = _positive_float(self.learning_rate, "cdan_learning_rate")
        weight_decay = _nonnegative_float(self.weight_decay, "cdan_weight_decay")
        cdan_loss_weight = _nonnegative_float(self.cdan_loss_weight, "cdan_loss_weight")
        domain_loss_weight = _nonnegative_float(self.domain_loss_weight, "cdan_domain_loss_weight")
        mmd_loss_weight = _nonnegative_float(self.mmd_loss_weight, "cdan_mmd_loss_weight")
        conditional_mmd_loss_weight = _nonnegative_float(
            self.conditional_mmd_loss_weight,
            "cdan_conditional_mmd_loss_weight",
        )
        cdan_randomized_dim = _optional_positive_int(self.cdan_randomized_dim, "cdan_randomized_dim")
        cdan_entropy_conditioning = _boolean(self.cdan_entropy_conditioning, "cdan_entropy_conditioning")
        mmd_kernel_mul = _positive_float(self.mmd_kernel_mul, "cdan_mmd_kernel_mul")
        mmd_kernel_num = _positive_int(self.mmd_kernel_num, "cdan_mmd_kernel_num")
        dropout = _bounded_float(self.dropout, "cdan_dropout", lower=0.0, upper=1.0)
        validation_fraction = _bounded_float(
            self.validation_fraction,
            "cdan_validation_fraction",
            lower=0.0,
            upper=1.0,
        )
        if cdan_loss_weight <= 0.0 and domain_loss_weight <= 0.0 and mmd_loss_weight <= 0.0 and conditional_mmd_loss_weight <= 0.0:
            raise ValueError("CDAN needs at least one positive adaptation loss weight.")

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
        model = _CDANModule(
            input_dim=x_source.shape[1],
            hidden_units=hidden_units,
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            dropout=dropout,
            randomized_dim=cdan_randomized_dim,
        ).to(device)

        if self.class_weight == "balanced":
            train_counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
            weights = train_idx.shape[0] / np.maximum(train_counts, 1.0) / float(n_classes)
            class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        else:
            class_weights = None
        class_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        domain_loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        source_tensor = torch.as_tensor(x_source, dtype=torch.float32, device=device)
        source_label_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
        target_tensor = torch.as_tensor(x_target, dtype=torch.float32, device=device)
        rng = np.random.default_rng(random_state)
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
                source_class_logits, source_embedding = model.classify(batch_source)
                target_class_logits, target_embedding = model.classify(batch_target)
                loss = class_loss_fn(source_class_logits, batch_source_labels)

                if cdan_loss_weight > 0.0:
                    loss = loss + cdan_loss_weight * _cdan_domain_loss(
                        model,
                        source_embedding,
                        target_embedding,
                        source_class_logits,
                        target_class_logits,
                        entropy_conditioning=cdan_entropy_conditioning,
                    )

                if domain_loss_weight > 0.0:
                    source_domain_logits = model.domain_logits(source_embedding, grl_scale=1.0)
                    target_domain_logits = model.domain_logits(target_embedding, grl_scale=1.0)
                    source_domain_labels = torch.zeros(source_domain_logits.shape[0], dtype=torch.long, device=device)
                    target_domain_labels = torch.ones(target_domain_logits.shape[0], dtype=torch.long, device=device)
                    domain_loss = 0.5 * (
                        domain_loss_fn(source_domain_logits, source_domain_labels)
                        + domain_loss_fn(target_domain_logits, target_domain_labels)
                    )
                    loss = loss + domain_loss_weight * domain_loss

                if mmd_loss_weight > 0.0:
                    loss = loss + mmd_loss_weight * _mmd_loss(
                        source_embedding,
                        target_embedding,
                        kernel_mul=mmd_kernel_mul,
                        kernel_num=mmd_kernel_num,
                    )

                if conditional_mmd_loss_weight > 0.0:
                    loss = loss + conditional_mmd_loss_weight * _conditional_mmd_loss(
                        source_embedding,
                        target_embedding,
                        batch_source_labels,
                        target_class_logits,
                        n_classes=n_classes,
                        kernel_mul=mmd_kernel_mul,
                        kernel_num=mmd_kernel_num,
                    )

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_logits, _ = model.classify(source_tensor[validation_idx])
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
        self.n_epochs_ = epochs_run
        self.best_source_validation_loss_ = float(best_loss)
        self.source_rows_ = int(x_source.shape[0])
        self.target_rows_ = int(x_target.shape[0])
        self.cdan_loss_weight_ = float(cdan_loss_weight)
        self.domain_loss_weight_ = float(domain_loss_weight)
        self.mmd_loss_weight_ = float(mmd_loss_weight)
        self.conditional_mmd_loss_weight_ = float(conditional_mmd_loss_weight)
        self.cdan_randomized_dim_ = None if cdan_randomized_dim is None else int(cdan_randomized_dim)
        self.cdan_input_dim_ = int(model.conditional_input_dim)
        self.cdan_entropy_conditioning_ = bool(cdan_entropy_conditioning)
        self.mmd_kernel_mul_ = float(mmd_kernel_mul)
        self.mmd_kernel_num_ = int(mmd_kernel_num)
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
            raise RuntimeError("TorchCDANClassifier must be fitted before prediction.")
        torch = _torch()
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_.classify(x)
        return logits.detach().cpu().numpy()

    def _protocol(self) -> str:
        cdan = float(getattr(self, "cdan_loss_weight_", self.cdan_loss_weight)) > 0.0
        domain = float(getattr(self, "domain_loss_weight_", self.domain_loss_weight)) > 0.0
        marginal = float(getattr(self, "mmd_loss_weight_", self.mmd_loss_weight)) > 0.0
        conditional = float(getattr(self, "conditional_mmd_loss_weight_", self.conditional_mmd_loss_weight)) > 0.0
        if cdan and not domain and not marginal and not conditional:
            return CDAN_PROTOCOL
        if cdan and not domain and marginal and not conditional:
            return CDAN_MMD_PROTOCOL
        if cdan and not domain and not marginal and conditional:
            return CDAN_CONDITIONAL_MMD_PROTOCOL
        if cdan and not domain and marginal and conditional:
            return CDAN_MMD_CONDITIONAL_MMD_PROTOCOL
        if cdan and domain and not marginal and not conditional:
            return CDAN_DANN_PROTOCOL
        if cdan and domain and marginal and not conditional:
            return CDAN_DANN_MMD_PROTOCOL
        if cdan and domain and not marginal and conditional:
            return CDAN_DANN_CONDITIONAL_MMD_PROTOCOL
        if cdan and domain and marginal and conditional:
            return CDAN_DANN_MMD_CONDITIONAL_MMD_PROTOCOL
        components: list[str] = []
        if domain:
            components.append("domain_adversarial")
        if marginal:
            components.append("mmd")
        if conditional:
            components.append("conditional_mmd")
        return "unlabeled_target_" + "_".join(components)

    def metadata(self) -> dict[str, Any]:
        cdan_loss_weight = float(getattr(self, "cdan_loss_weight_", self.cdan_loss_weight))
        domain_loss_weight = float(getattr(self, "domain_loss_weight_", self.domain_loss_weight))
        mmd_loss_weight = float(getattr(self, "mmd_loss_weight_", self.mmd_loss_weight))
        conditional_mmd_loss_weight = float(getattr(self, "conditional_mmd_loss_weight_", self.conditional_mmd_loss_weight))
        cdan_randomized_dim = getattr(self, "cdan_randomized_dim_", self.cdan_randomized_dim)
        return {
            "cdan_adaptation": True,
            "cdan_protocol": self._protocol(),
            "cdan_uses_target_features": True,
            "cdan_uses_target_labels": False,
            "cdan_valid_for_benchmark": True,
            "cdan_hidden_units": int(self.hidden_units),
            "cdan_embedding_dim": int(self.embedding_dim),
            "cdan_max_epochs": int(self.max_epochs),
            "cdan_epochs_run": int(getattr(self, "n_epochs_", 0)),
            "cdan_batch_size": int(self.batch_size),
            "cdan_learning_rate": float(self.learning_rate),
            "cdan_weight_decay": float(self.weight_decay),
            "cdan_loss_weight": cdan_loss_weight,
            "cdan_domain_loss_weight": domain_loss_weight,
            "cdan_mmd_loss_weight": mmd_loss_weight,
            "cdan_conditional_mmd_loss_weight": conditional_mmd_loss_weight,
            "cdan_mmd_adaptation": bool(mmd_loss_weight > 0.0 or conditional_mmd_loss_weight > 0.0),
            "cdan_randomized_dim": "" if cdan_randomized_dim is None else int(cdan_randomized_dim),
            "cdan_input_dim": int(getattr(self, "cdan_input_dim_", 0)),
            "cdan_entropy_conditioning": bool(getattr(self, "cdan_entropy_conditioning_", self.cdan_entropy_conditioning)),
            "cdan_mmd_kernel_mul": float(getattr(self, "mmd_kernel_mul_", self.mmd_kernel_mul)),
            "cdan_mmd_kernel_num": int(getattr(self, "mmd_kernel_num_", self.mmd_kernel_num)),
            "cdan_patience": int(self.patience),
            "cdan_dropout": float(self.dropout),
            "cdan_source_rows": int(getattr(self, "source_rows_", 0)),
            "cdan_target_rows": int(getattr(self, "target_rows_", 0)),
            "cdan_source_validation_loss": float(getattr(self, "best_source_validation_loss_", np.nan)),
            "cdan_device": str(getattr(self, "device_", self.device)),
        }


class _CDANModule:
    def __new__(
        cls,
        *,
        input_dim: int,
        hidden_units: int,
        embedding_dim: int,
        n_classes: int,
        dropout: float,
        randomized_dim: int | None,
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
                if randomized_dim is None:
                    self.randomized_cdan = False
                    self.conditional_input_dim = int(embedding_dim * n_classes)
                else:
                    self.randomized_cdan = True
                    self.conditional_input_dim = int(randomized_dim)
                    feature_projection = torch.randn(embedding_dim, randomized_dim) / max(float(embedding_dim), 1.0) ** 0.5
                    probability_projection = torch.randn(n_classes, randomized_dim) / max(float(n_classes), 1.0) ** 0.5
                    self.register_buffer("feature_projection", feature_projection)
                    self.register_buffer("probability_projection", probability_projection)
                self.conditional_domain_head = torch.nn.Sequential(
                    torch.nn.Linear(self.conditional_input_dim, hidden_units),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_units, 2),
                )

            def classify(self, features):
                embedding = self.feature_extractor(features)
                return self.class_head(embedding), embedding

            def domain_logits(self, embedding, *, grl_scale: float):
                return self.domain_head(_GradientReverse.apply(embedding, float(grl_scale)))

            def conditional_domain_logits(self, embedding, probabilities, *, grl_scale: float):
                conditional_features = _conditional_domain_features(self, embedding, probabilities)
                return self.conditional_domain_head(_GradientReverse.apply(conditional_features, float(grl_scale)))

        return Module()


def _conditional_domain_features(model, embedding, probabilities):
    torch = _torch()
    if bool(getattr(model, "randomized_cdan", False)):
        projected_embedding = embedding @ model.feature_projection
        projected_probabilities = probabilities @ model.probability_projection
        return (projected_embedding * projected_probabilities) / torch.sqrt(
            torch.as_tensor(float(model.conditional_input_dim), dtype=embedding.dtype, device=embedding.device)
        )
    outer = torch.bmm(probabilities.unsqueeze(2), embedding.unsqueeze(1))
    return outer.reshape(embedding.shape[0], -1)


def _entropy_weights(probabilities):
    torch = _torch()
    entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-6))).sum(dim=1)
    weights = 1.0 + torch.exp(-entropy)
    return weights / torch.clamp(weights.detach().mean(), min=1e-6)


def _weighted_mean(losses, weights):
    torch = _torch()
    weights = weights.to(dtype=losses.dtype, device=losses.device).reshape(-1)
    return (losses.reshape(-1) * weights).sum() / torch.clamp(weights.sum(), min=1e-6)


def _cdan_domain_loss(
    model,
    source_embedding,
    target_embedding,
    source_class_logits,
    target_class_logits,
    *,
    entropy_conditioning: bool,
):
    torch = _torch()
    source_probabilities = torch.softmax(source_class_logits, dim=1)
    target_probabilities = torch.softmax(target_class_logits, dim=1)
    source_domain_logits = model.conditional_domain_logits(source_embedding, source_probabilities, grl_scale=1.0)
    target_domain_logits = model.conditional_domain_logits(target_embedding, target_probabilities, grl_scale=1.0)
    source_domain_labels = torch.zeros(source_domain_logits.shape[0], dtype=torch.long, device=source_domain_logits.device)
    target_domain_labels = torch.ones(target_domain_logits.shape[0], dtype=torch.long, device=target_domain_logits.device)
    source_losses = torch.nn.functional.cross_entropy(source_domain_logits, source_domain_labels, reduction="none")
    target_losses = torch.nn.functional.cross_entropy(target_domain_logits, target_domain_labels, reduction="none")
    if entropy_conditioning:
        source_loss = _weighted_mean(source_losses, _entropy_weights(source_probabilities))
        target_loss = _weighted_mean(target_losses, _entropy_weights(target_probabilities))
    else:
        source_loss = source_losses.mean()
        target_loss = target_losses.mean()
    return 0.5 * (source_loss + target_loss)


def fit_cdan_predict_proba(
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
    cdan_loss_weight: float = 0.1,
    domain_loss_weight: float = 0.0,
    mmd_loss_weight: float = 0.0,
    conditional_mmd_loss_weight: float = 0.0,
    cdan_randomized_dim: int | None = None,
    cdan_entropy_conditioning: bool = True,
    mmd_kernel_mul: float = 2.0,
    mmd_kernel_num: int = 5,
    validation_fraction: float = 0.1,
    patience: int = 10,
    dropout: float = 0.1,
    random_state: int | None = 13,
    device: str = "auto",
) -> CDANFitResult:
    model = TorchCDANClassifier(
        hidden_units=hidden_units,
        embedding_dim=embedding_dim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        cdan_loss_weight=cdan_loss_weight,
        domain_loss_weight=domain_loss_weight,
        mmd_loss_weight=mmd_loss_weight,
        conditional_mmd_loss_weight=conditional_mmd_loss_weight,
        cdan_randomized_dim=cdan_randomized_dim,
        cdan_entropy_conditioning=cdan_entropy_conditioning,
        mmd_kernel_mul=mmd_kernel_mul,
        mmd_kernel_num=mmd_kernel_num,
        validation_fraction=validation_fraction,
        patience=patience,
        dropout=dropout,
        random_state=random_state,
        device=device,
    )
    model.fit(source_features, source_labels, target_features=target_features)
    probabilities = model.predict_proba(target_features)
    return CDANFitResult(model=model, probabilities=probabilities, metadata=model.metadata())
