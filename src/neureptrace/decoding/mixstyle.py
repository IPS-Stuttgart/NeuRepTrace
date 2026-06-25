"""Source-only MixStyle feature augmentation for cross-subject decoding.

MixStyle-style augmentation perturbs source feature rows by mixing per-domain
feature statistics between source subjects/domains.  This module implements a
feature-matrix variant that is deliberately Category 1: it uses only source
features, source labels, and source domain ids.  Held-out target features and
held-out target labels are not accepted by the public API.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MIXSTYLE_PROTOCOL = "source_only_mixstyle_feature_augmentation"
SOURCE_MIXSTYLE_CATEGORY = "1_strict_source_only"
DEFAULT_MIXSTYLE_ALPHA = 0.2
DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW = 1
DEFAULT_MIXSTYLE_RANDOM_STATE = 13
_MIN_SCALE = 1.0e-12


@dataclass(frozen=True, slots=True)
class SourceMixStyleConfig:
    """Configuration for source-only feature-statistics mixing."""

    augmentations_per_row: int = DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW
    alpha: float = DEFAULT_MIXSTYLE_ALPHA
    random_state: int | None = DEFAULT_MIXSTYLE_RANDOM_STATE
    domain_pairing: str = "shuffle"
    preserve_domain_mean: bool = False
    class_conditional: bool = False
    include_original: bool = True


@dataclass(frozen=True, slots=True)
class SourceMixStyleResult:
    """Augmented source features, labels, domains, and provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    synthetic_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_original(self) -> int:
        """Number of original source rows retained in ``features``."""

        return int(np.count_nonzero(~self.synthetic_mask))

    @property
    def n_synthetic(self) -> int:
        """Number of synthetic MixStyle rows in ``features``."""

        return int(np.count_nonzero(self.synthetic_mask))


# pylint: disable-next=too-many-arguments,too-many-locals

def augment_source_mixstyle(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    *,
    augmentations_per_row: int | str = DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW,
    alpha: float | str = DEFAULT_MIXSTYLE_ALPHA,
    random_state: int | str | None = DEFAULT_MIXSTYLE_RANDOM_STATE,
    domain_pairing: str = "shuffle",
    preserve_domain_mean: bool = False,
    class_conditional: bool = False,
    include_original: bool = True,
) -> SourceMixStyleResult:
    """Append source-only MixStyle synthetic rows.

    Parameters
    ----------
    source_features:
        Source feature matrix, typically rows from all source subjects in a LOSO
        fold.
    source_labels:
        One source label per source row.  Synthetic rows keep the label of the
        original row being stylized.
    source_domains:
        One source-domain identifier per row, usually subject id.
    augmentations_per_row:
        Number of synthetic rows to create for each source row.
    alpha:
        Beta-distribution concentration for the style interpolation coefficient.
        Smaller values produce more nearly one-domain statistics; larger values
        produce more evenly mixed statistics.
    random_state:
        Optional deterministic RNG seed.
    domain_pairing:
        ``"shuffle"`` samples partner domains randomly, ``"nearest"`` chooses the
        most similar source-domain statistics, and ``"farthest"`` chooses the most
        dissimilar source-domain statistics.
    preserve_domain_mean:
        If true, only feature scales are mixed and the original row's domain mean
        is retained.  If false, both mean and scale are mixed.
    class_conditional:
        If true, partner-domain statistics are estimated from rows with the same
        source label when possible.  This still uses source labels only.
    include_original:
        Whether to prepend the original rows before synthetic rows.

    Returns
    -------
    SourceMixStyleResult
        Augmented feature matrix, labels, domain ids, synthetic mask, and protocol
        metadata.

    Notes
    -----
    This is a strict source-only protocol.  The function has no target-feature or
    target-label arguments, and all metadata mark it as valid for Category 1.
    """

    features = _feature_matrix(source_features, name="source_features")
    labels = _label_vector(source_labels, expected_length=features.shape[0])
    domains = _domain_vector(source_domains, expected_length=features.shape[0])
    n_aug = _normalize_nonnegative_int(augmentations_per_row, name="augmentations_per_row")
    beta_alpha = _positive_float(alpha, name="alpha")
    pairing = normalize_mixstyle_domain_pairing(domain_pairing)
    seed = None if random_state in {None, "", "none", "None"} else _normalize_nonnegative_int(random_state, name="random_state")
    rng = np.random.default_rng(seed)

    domain_stats = _domain_statistics(features, domains)
    domain_names = tuple(domain_stats)
    if len(domain_names) < 2 and n_aug > 0:
        raise ValueError("Source MixStyle requires at least two source domains when augmentations_per_row > 0.")
    label_domain_stats = _label_domain_statistics(features, labels, domains) if class_conditional else {}
    deterministic_partner_order = _deterministic_partner_order(domain_stats)

    synthetic_features: list[np.ndarray] = []
    synthetic_labels: list[Any] = []
    synthetic_domains: list[str] = []
    for row, label, domain in zip(features, labels, domains, strict=True):
        for _ in range(n_aug):
            partner = _choose_partner_domain(
                domain,
                domain_names=domain_names,
                deterministic_partner_order=deterministic_partner_order,
                pairing=pairing,
                rng=rng,
            )
            source_stats = _stats_for_row(label_domain_stats, domain_stats, label, domain)
            partner_stats = _stats_for_row(label_domain_stats, domain_stats, label, partner)
            mix = float(rng.beta(beta_alpha, beta_alpha))
            synthetic = _mixstyle_row(
                row,
                source_stats=source_stats,
                partner_stats=partner_stats,
                mix=mix,
                preserve_domain_mean=bool(preserve_domain_mean),
            )
            synthetic_features.append(synthetic)
            synthetic_labels.append(label)
            synthetic_domains.append(f"mixstyle:{domain}->{partner}")

    feature_blocks = []
    label_blocks = []
    domain_blocks = []
    mask_blocks = []
    if include_original:
        feature_blocks.append(features)
        label_blocks.append(labels)
        domain_blocks.append(domains)
        mask_blocks.append(np.zeros(features.shape[0], dtype=bool))
    if synthetic_features:
        feature_blocks.append(np.vstack(synthetic_features))
        label_blocks.append(np.asarray(synthetic_labels, dtype=object))
        domain_blocks.append(np.asarray(synthetic_domains, dtype=object))
        mask_blocks.append(np.ones(len(synthetic_features), dtype=bool))
    if not feature_blocks:
        raise ValueError("No rows would be returned; enable include_original or request augmentations_per_row > 0.")

    output_features = np.vstack(feature_blocks).astype(np.float32, copy=False)
    output_labels = np.concatenate(label_blocks).astype(object, copy=False)
    output_domains = np.concatenate(domain_blocks).astype(object, copy=False)
    synthetic_mask = np.concatenate(mask_blocks)
    metadata = _metadata(
        n_source_rows=features.shape[0],
        n_output_rows=output_features.shape[0],
        n_synthetic_rows=int(np.count_nonzero(synthetic_mask)),
        n_domains=len(domain_names),
        n_classes=int(np.unique(labels).shape[0]),
        feature_dim=features.shape[1],
        augmentations_per_row=n_aug,
        alpha=beta_alpha,
        random_state=seed,
        domain_pairing=pairing,
        preserve_domain_mean=bool(preserve_domain_mean),
        class_conditional=bool(class_conditional),
        include_original=bool(include_original),
    )
    return SourceMixStyleResult(
        features=output_features,
        labels=output_labels,
        domains=output_domains,
        synthetic_mask=synthetic_mask,
        metadata=metadata,
    )


def source_mixstyle_config(
    *,
    augmentations_per_row: int | str = DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW,
    alpha: float | str = DEFAULT_MIXSTYLE_ALPHA,
    random_state: int | str | None = DEFAULT_MIXSTYLE_RANDOM_STATE,
    domain_pairing: str = "shuffle",
    preserve_domain_mean: bool = False,
    class_conditional: bool = False,
    include_original: bool = True,
) -> SourceMixStyleConfig:
    """Normalize user-facing MixStyle options."""

    return SourceMixStyleConfig(
        augmentations_per_row=_normalize_nonnegative_int(augmentations_per_row, name="augmentations_per_row"),
        alpha=_positive_float(alpha, name="alpha"),
        random_state=None if random_state in {None, "", "none", "None"} else _normalize_nonnegative_int(random_state, name="random_state"),
        domain_pairing=normalize_mixstyle_domain_pairing(domain_pairing),
        preserve_domain_mean=bool(preserve_domain_mean),
        class_conditional=bool(class_conditional),
        include_original=bool(include_original),
    )


def augment_source_mixstyle_from_config(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    config: SourceMixStyleConfig | Mapping[str, Any] | None = None,
) -> SourceMixStyleResult:
    """Apply MixStyle from a dataclass or mapping configuration."""

    cfg = source_mixstyle_config() if config is None else config
    if isinstance(cfg, Mapping):
        cfg = source_mixstyle_config(**dict(cfg))
    return augment_source_mixstyle(
        source_features,
        source_labels,
        source_domains,
        augmentations_per_row=cfg.augmentations_per_row,
        alpha=cfg.alpha,
        random_state=cfg.random_state,
        domain_pairing=cfg.domain_pairing,
        preserve_domain_mean=cfg.preserve_domain_mean,
        class_conditional=cfg.class_conditional,
        include_original=cfg.include_original,
    )


def normalize_mixstyle_domain_pairing(value: str | None) -> str:
    """Normalize MixStyle partner-domain selection mode."""

    normalized = "shuffle" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "random": "shuffle",
        "sample": "shuffle",
        "nearest_domain": "nearest",
        "similar": "nearest",
        "most_similar": "nearest",
        "farthest_domain": "farthest",
        "distant": "farthest",
        "most_different": "farthest",
    }.get(normalized, normalized)
    if normalized not in {"shuffle", "nearest", "farthest"}:
        raise ValueError("domain_pairing must be one of: shuffle, nearest, farthest.")
    return normalized


def _mixstyle_row(
    row: np.ndarray,
    *,
    source_stats: tuple[np.ndarray, np.ndarray],
    partner_stats: tuple[np.ndarray, np.ndarray],
    mix: float,
    preserve_domain_mean: bool,
) -> np.ndarray:
    source_mean, source_std = source_stats
    partner_mean, partner_std = partner_stats
    normalized = (row - source_mean) / np.maximum(source_std, _MIN_SCALE)
    mixed_std = mix * source_std + (1.0 - mix) * partner_std
    mixed_mean = source_mean if preserve_domain_mean else mix * source_mean + (1.0 - mix) * partner_mean
    return normalized * mixed_std + mixed_mean


def _domain_statistics(features: np.ndarray, domains: np.ndarray) -> dict[Hashable, tuple[np.ndarray, np.ndarray]]:
    stats = {}
    for domain in dict.fromkeys(domains.tolist()):
        rows = features[domains == domain]
        stats[domain] = _mean_std(rows)
    return stats


def _label_domain_statistics(features: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> dict[tuple[Any, Hashable], tuple[np.ndarray, np.ndarray]]:
    stats = {}
    for label in dict.fromkeys(labels.tolist()):
        for domain in dict.fromkeys(domains.tolist()):
            mask = (labels == label) & (domains == domain)
            if np.any(mask):
                stats[(label, domain)] = _mean_std(features[mask])
    return stats


def _stats_for_row(
    label_domain_stats: Mapping[tuple[Any, Hashable], tuple[np.ndarray, np.ndarray]],
    domain_stats: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
    label: Any,
    domain: Hashable,
) -> tuple[np.ndarray, np.ndarray]:
    return label_domain_stats.get((label, domain), domain_stats[domain])


def _mean_std(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(rows, axis=0)
    std = np.std(rows, axis=0, ddof=1 if rows.shape[0] > 1 else 0)
    std = np.maximum(std, _MIN_SCALE)
    return mean, std


def _deterministic_partner_order(domain_stats: Mapping[Hashable, tuple[np.ndarray, np.ndarray]]) -> dict[Hashable, tuple[Hashable, ...]]:
    order = {}
    domains = tuple(domain_stats)
    for domain in domains:
        domain_mean, domain_std = domain_stats[domain]
        distances = []
        for other in domains:
            if other == domain:
                continue
            other_mean, other_std = domain_stats[other]
            distance = float(np.linalg.norm(domain_mean - other_mean) + np.linalg.norm(domain_std - other_std))
            distances.append((distance, repr(other), other))
        order[domain] = tuple(item[2] for item in sorted(distances))
    return order


def _choose_partner_domain(
    domain: Hashable,
    *,
    domain_names: tuple[Hashable, ...],
    deterministic_partner_order: Mapping[Hashable, tuple[Hashable, ...]],
    pairing: str,
    rng: np.random.Generator,
) -> Hashable:
    candidates = tuple(candidate for candidate in domain_names if candidate != domain)
    if not candidates:
        raise ValueError("At least two source domains are required to choose a MixStyle partner.")
    if pairing == "shuffle":
        return candidates[int(rng.integers(0, len(candidates)))]
    ordered = deterministic_partner_order[domain]
    return ordered[0] if pairing == "nearest" else ordered[-1]


def _metadata(
    *,
    n_source_rows: int,
    n_output_rows: int,
    n_synthetic_rows: int,
    n_domains: int,
    n_classes: int,
    feature_dim: int,
    augmentations_per_row: int,
    alpha: float,
    random_state: int | None,
    domain_pairing: str,
    preserve_domain_mean: bool,
    class_conditional: bool,
    include_original: bool,
) -> dict[str, Any]:
    return {
        "source_mixstyle_augmentation": True,
        "source_mixstyle_protocol": SOURCE_MIXSTYLE_PROTOCOL,
        "source_mixstyle_protocol_category": SOURCE_MIXSTYLE_CATEGORY,
        "source_mixstyle_uses_source_features": True,
        "source_mixstyle_uses_source_labels": True,
        "source_mixstyle_uses_source_domains": True,
        "source_mixstyle_uses_target_features": False,
        "source_mixstyle_uses_target_labels": False,
        "source_mixstyle_valid_for_strict_source_only": True,
        "source_mixstyle_valid_for_unlabeled_target_adaptation": True,
        "source_mixstyle_debug_upper_bound": False,
        "source_mixstyle_n_source_rows": int(n_source_rows),
        "source_mixstyle_n_output_rows": int(n_output_rows),
        "source_mixstyle_n_synthetic_rows": int(n_synthetic_rows),
        "source_mixstyle_n_domains": int(n_domains),
        "source_mixstyle_n_classes": int(n_classes),
        "source_mixstyle_feature_dim": int(feature_dim),
        "source_mixstyle_augmentations_per_row": int(augmentations_per_row),
        "source_mixstyle_alpha": float(alpha),
        "source_mixstyle_random_state": "" if random_state is None else int(random_state),
        "source_mixstyle_domain_pairing": domain_pairing,
        "source_mixstyle_preserve_domain_mean": bool(preserve_domain_mean),
        "source_mixstyle_class_conditional": bool(class_conditional),
        "source_mixstyle_include_original": bool(include_original),
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


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_labels must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    for domain in vector.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {domain!r}.") from exc
    return vector


def _normalize_nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def _normalize_nonnegative_int_or_none(value: int | str | None, *, name: str) -> int | None:
    if value is None:
        return None
    return _normalize_nonnegative_int(value, name=name)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric
