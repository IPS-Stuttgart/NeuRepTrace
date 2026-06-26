"""Source-only MixUp feature augmentation for domain generalization.

The helpers in this module implement dependency-light feature-space MixUp for
cross-subject M/EEG decoding.  Synthetic rows are convex combinations of source
rows.  Labels are represented both as hard labels for existing scikit-learn style
pipelines and as class-probability targets for consumers that support soft-label
training.

This is a strict source-only / Protocol-1 utility: the public APIs use source
features, source labels, and optional source-domain ids only.  Target features and
target labels are intentionally not accepted.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MIXUP_AUGMENTATION = "source_mixup"
SOURCE_MIXUP_PROTOCOL = "strict_source_only_mixup_augmentation"
SOURCE_MIXUP_CATEGORY = "1_strict_source_only"
HARD_LABEL_POLICIES = ("content", "partner", "dominant")
DEFAULT_MIXUP_ALPHA = 0.4


@dataclass(frozen=True, slots=True)
class SourceMixUpConfig:
    """Configuration for source-only feature-space MixUp."""

    synthetic_per_class: int = 0
    alpha: float = DEFAULT_MIXUP_ALPHA
    random_state: int | None = 13
    same_class_partner: bool = True
    cross_domain_partner: bool = True
    hard_label_policy: str = "content"
    preserve_original: bool = True

    @property
    def enabled(self) -> bool:
        """Whether this config requests synthetic rows."""

        return self.synthetic_per_class > 0


@dataclass(frozen=True, slots=True)
class SourceMixUpResult:
    """Augmented features, hard labels, soft labels, and provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    label_distributions: np.ndarray
    synthetic_mask: np.ndarray
    content_indices: np.ndarray
    partner_indices: np.ndarray
    lambdas: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_synthetic(self) -> int:
        """Number of synthetic rows in the output."""

        return int(np.sum(self.synthetic_mask))


# pylint: disable-next=too-many-arguments,too-many-locals

def augment_source_with_mixup(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    *,
    source_domains: Sequence[Hashable] | np.ndarray | None = None,
    config: SourceMixUpConfig | Mapping[str, Any] | None = None,
) -> SourceMixUpResult:
    """Append source-only MixUp synthetic rows.

    Parameters
    ----------
    source_features:
        Source feature matrix.  Rows are trials/windows; columns are flattened
        M/EEG features or latent features.
    source_labels:
        One source class label per row.
    source_domains:
        Optional source-domain identifiers, usually source-subject ids.  When
        ``cross_domain_partner=True``, partner rows are drawn from another source
        domain whenever possible.
    config:
        MixUp settings.  A mapping is normalized through
        :func:`source_mixup_config`.

    Returns
    -------
    SourceMixUpResult
        Feature rows, hard labels, soft label distributions, synthetic mask, and
        protocol metadata.

    Notes
    -----
    The API intentionally has no target-feature or target-label arguments.  This
    keeps the method valid for strict source-only Protocol 1 evaluation.
    """

    cfg = source_mixup_config() if config is None else _coerce_config(config)
    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0], name="source_labels")
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=labels.dtype if labels.dtype != object else object)
    class_to_index = {class_label: index for index, class_label in enumerate(classes.tolist())}
    original_distributions = _one_hot(labels, class_to_index)

    if not cfg.enabled:
        metadata = _metadata(cfg, n_source_rows=features.shape[0], n_synthetic_rows=0, n_classes=classes.shape[0], n_source_domains=np.unique(domains).shape[0])
        return SourceMixUpResult(
            features=features.astype(np.float32, copy=False),
            labels=labels.copy(),
            classes=classes,
            label_distributions=original_distributions,
            synthetic_mask=np.zeros(features.shape[0], dtype=bool),
            content_indices=np.empty(0, dtype=int),
            partner_indices=np.empty(0, dtype=int),
            lambdas=np.empty(0, dtype=float),
            metadata=metadata,
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    synthetic_distributions: list[np.ndarray] = []
    content_indices: list[int] = []
    partner_indices: list[int] = []
    lambdas: list[float] = []

    for class_label in classes.tolist():
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size == 0:
            continue
        for _ in range(cfg.synthetic_per_class):
            content_index = int(rng.choice(class_indices))
            partner_pool = _partner_pool(
                labels,
                domains,
                content_index=content_index,
                class_indices=class_indices,
                same_class_partner=cfg.same_class_partner,
                cross_domain_partner=cfg.cross_domain_partner,
            )
            partner_index = int(rng.choice(partner_pool))
            lam = float(rng.beta(cfg.alpha, cfg.alpha))
            row = mixup_rows(
                features[content_index : content_index + 1],
                features[partner_index : partner_index + 1],
                lambdas=np.asarray([lam], dtype=float),
            )[0]
            distribution = np.zeros(classes.shape[0], dtype=float)
            distribution[class_to_index[labels[content_index]]] += lam
            distribution[class_to_index[labels[partner_index]]] += 1.0 - lam
            synthetic_rows.append(row)
            synthetic_distributions.append(distribution)
            synthetic_labels.append(_hard_label(labels[content_index], labels[partner_index], lam, policy=cfg.hard_label_policy))
            content_indices.append(content_index)
            partner_indices.append(partner_index)
            lambdas.append(lam)

    synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
    synthetic_label_array = np.asarray(synthetic_labels, dtype=labels.dtype if labels.dtype != object else object)
    synthetic_distribution_array = np.vstack(synthetic_distributions).astype(np.float32, copy=False) if synthetic_distributions else np.empty((0, classes.shape[0]), dtype=np.float32)

    if cfg.preserve_original:
        output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
        output_labels = np.concatenate([labels, synthetic_label_array])
        output_distributions = np.vstack([original_distributions, synthetic_distribution_array]).astype(np.float32, copy=False)
        synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    else:
        output_features = synthetic_features
        output_labels = synthetic_label_array
        output_distributions = synthetic_distribution_array
        synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    metadata = _metadata(
        cfg,
        n_source_rows=features.shape[0],
        n_synthetic_rows=synthetic_features.shape[0],
        n_classes=classes.shape[0],
        n_source_domains=np.unique(domains).shape[0],
    )
    return SourceMixUpResult(
        features=output_features,
        labels=output_labels,
        classes=classes,
        label_distributions=output_distributions,
        synthetic_mask=synthetic_mask,
        content_indices=np.asarray(content_indices, dtype=int),
        partner_indices=np.asarray(partner_indices, dtype=int),
        lambdas=np.asarray(lambdas, dtype=float),
        metadata=metadata,
    )


def mixup_rows(
    content_features: Sequence[Sequence[float]] | np.ndarray,
    partner_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    lambdas: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Return convex combinations of content and partner feature rows."""

    content = _feature_matrix(content_features, name="content_features")
    partner = _feature_matrix(partner_features, name="partner_features")
    if content.shape != partner.shape:
        raise ValueError(f"content_features and partner_features must have the same shape: {content.shape} != {partner.shape}.")
    lam = np.asarray(lambdas, dtype=float)
    if lam.ndim == 0:
        lam = np.full(content.shape[0], float(lam), dtype=float)
    lam = lam.reshape(-1, 1)
    if lam.shape[0] != content.shape[0]:
        raise ValueError(f"lambdas must be scalar or contain one value per row: {lam.shape[0]} != {content.shape[0]}.")
    if not np.all(np.isfinite(lam)) or np.any(lam < 0.0) or np.any(lam > 1.0):
        raise ValueError("lambdas must be finite values in [0, 1].")
    return (lam * content + (1.0 - lam) * partner).astype(np.float32, copy=False)


def source_mixup_config(
    *,
    synthetic_per_class: int | str = 0,
    alpha: float | str = DEFAULT_MIXUP_ALPHA,
    random_state: int | str | None = 13,
    same_class_partner: bool = True,
    cross_domain_partner: bool = True,
    hard_label_policy: str = "content",
    preserve_original: bool = True,
) -> SourceMixUpConfig:
    """Normalize user-facing source-MixUp options."""

    return SourceMixUpConfig(
        synthetic_per_class=_normalize_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        alpha=_positive_float(alpha, name="alpha"),
        random_state=None if random_state in {None, "", "none", "None"} else _normalize_integer(random_state, name="random_state"),
        same_class_partner=bool(same_class_partner),
        cross_domain_partner=bool(cross_domain_partner),
        hard_label_policy=normalize_hard_label_policy(hard_label_policy),
        preserve_original=bool(preserve_original),
    )


def normalize_hard_label_policy(value: str | None) -> str:
    """Normalize hard-label policies for synthetic MixUp rows."""

    normalized = "content" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "content_label": "content",
        "source": "content",
        "partner_label": "partner",
        "style": "partner",
        "lambda_dominant": "dominant",
        "majority": "dominant",
        "argmax": "dominant",
    }.get(normalized, normalized)
    if normalized not in HARD_LABEL_POLICIES:
        raise ValueError(f"Unknown hard_label_policy {value!r}. Available policies: {', '.join(HARD_LABEL_POLICIES)}.")
    return normalized


def _coerce_config(config: SourceMixUpConfig | Mapping[str, Any]) -> SourceMixUpConfig:
    if isinstance(config, SourceMixUpConfig):
        return config
    return source_mixup_config(**dict(config))


def _partner_pool(
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    content_index: int,
    class_indices: np.ndarray,
    same_class_partner: bool,
    cross_domain_partner: bool,
) -> np.ndarray:
    pool = class_indices.copy() if same_class_partner else np.arange(labels.shape[0], dtype=int)
    if pool.size > 1:
        pool = pool[pool != content_index]
    if cross_domain_partner:
        cross_pool = pool[domains[pool] != domains[content_index]]
        if cross_pool.size:
            pool = cross_pool
    if pool.size == 0:
        pool = np.asarray([content_index], dtype=int)
    return pool


def _hard_label(content_label: Any, partner_label: Any, lam: float, *, policy: str) -> Any:
    if policy == "content":
        return content_label
    if policy == "partner":
        return partner_label
    if policy == "dominant":
        return content_label if lam >= 0.5 else partner_label
    raise ValueError(f"Unhandled hard-label policy {policy!r}.")


def _one_hot(labels: np.ndarray, class_to_index: Mapping[Any, int]) -> np.ndarray:
    distributions = np.zeros((labels.shape[0], len(class_to_index)), dtype=np.float32)
    for row, label in enumerate(labels.tolist()):
        distributions[row, class_to_index[label]] = 1.0
    return distributions


def _metadata(cfg: SourceMixUpConfig, *, n_source_rows: int, n_synthetic_rows: int, n_classes: int, n_source_domains: int) -> dict[str, Any]:
    return {
        "source_mixup": bool(cfg.enabled),
        "source_mixup_protocol": SOURCE_MIXUP_PROTOCOL,
        "source_mixup_protocol_category": SOURCE_MIXUP_CATEGORY,
        "source_mixup_method": SOURCE_MIXUP_AUGMENTATION,
        "source_mixup_uses_source_features": True,
        "source_mixup_uses_source_labels": True,
        "source_mixup_uses_source_domains": True,
        "source_mixup_uses_target_features": False,
        "source_mixup_uses_target_labels": False,
        "source_mixup_valid_for_strict_source_only": True,
        "source_mixup_valid_for_unlabeled_target_adaptation": True,
        "source_mixup_valid_for_benchmark": True,
        "source_mixup_soft_label_distributions_available": True,
        "source_mixup_n_source_rows": int(n_source_rows),
        "source_mixup_n_synthetic_rows": int(n_synthetic_rows),
        "source_mixup_n_output_rows": int(n_source_rows + n_synthetic_rows if cfg.preserve_original else n_synthetic_rows),
        "source_mixup_n_classes": int(n_classes),
        "source_mixup_n_source_domains": int(n_source_domains),
        "source_mixup_synthetic_per_class": int(cfg.synthetic_per_class),
        "source_mixup_alpha": float(cfg.alpha),
        "source_mixup_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_mixup_same_class_partner": bool(cfg.same_class_partner),
        "source_mixup_cross_domain_partner": bool(cfg.cross_domain_partner),
        "source_mixup_hard_label_policy": cfg.hard_label_policy,
        "source_mixup_preserve_original": bool(cfg.preserve_original),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray | None, *, expected_length: int) -> np.ndarray:
    if values is None:
        return np.full(expected_length, "source", dtype=object)
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per feature row: {vector.shape[0]} != {expected_length}.")
    for value in vector.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return vector


def _normalize_integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _normalize_nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _normalize_integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
