"""Feature-space generative augmentation for source-only and target-adaptive decoding.

The utilities in this module operate on already extracted feature matrices.  The
Gaussian modes remain dependency-light baselines; the GAN and diffusion modes use
the optional ``torch`` extra to train fold-local neural generators.  All modes are
protocol-explicit:

* ``source_gaussian`` / ``source_gan`` / ``source_diffusion`` synthesize rows from
  source training rows and source labels only.  This is strict source-only
  augmentation.
* ``target_style_gaussian`` / ``target_style_gan`` / ``target_style_diffusion``
  synthesize class-conditional source rows and then match them to unlabeled target
  feature statistics.  This is category-2 target-adaptive augmentation and must be
  reported separately from strict source-only results.
* ``target_calibrated_gaussian`` / ``target_calibrated_gan`` /
  ``target_calibrated_diffusion`` may use disjoint target calibration rows and
  labels.  This is category-3 calibrated augmentation and must not be used as a
  zero-calibration benchmark.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SOURCE_GAUSSIAN_AUGMENTATION = "source_gaussian"
TARGET_STYLE_GAUSSIAN_AUGMENTATION = "target_style_gaussian"
TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION = "target_calibrated_gaussian"

SOURCE_GAN_AUGMENTATION = "source_gan"
TARGET_STYLE_GAN_AUGMENTATION = "target_style_gan"
TARGET_CALIBRATED_GAN_AUGMENTATION = "target_calibrated_gan"

SOURCE_DIFFUSION_AUGMENTATION = "source_diffusion"
TARGET_STYLE_DIFFUSION_AUGMENTATION = "target_style_diffusion"
TARGET_CALIBRATED_DIFFUSION_AUGMENTATION = "target_calibrated_diffusion"

SOURCE_ONLY_AUGMENTATIONS = (
    SOURCE_GAUSSIAN_AUGMENTATION,
    SOURCE_GAN_AUGMENTATION,
    SOURCE_DIFFUSION_AUGMENTATION,
)
TARGET_STYLE_AUGMENTATIONS = (
    TARGET_STYLE_GAUSSIAN_AUGMENTATION,
    TARGET_STYLE_GAN_AUGMENTATION,
    TARGET_STYLE_DIFFUSION_AUGMENTATION,
)
TARGET_CALIBRATED_AUGMENTATIONS = (
    TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
    TARGET_CALIBRATED_GAN_AUGMENTATION,
    TARGET_CALIBRATED_DIFFUSION_AUGMENTATION,
)
GAN_AUGMENTATIONS = (
    SOURCE_GAN_AUGMENTATION,
    TARGET_STYLE_GAN_AUGMENTATION,
    TARGET_CALIBRATED_GAN_AUGMENTATION,
)
DIFFUSION_AUGMENTATIONS = (
    SOURCE_DIFFUSION_AUGMENTATION,
    TARGET_STYLE_DIFFUSION_AUGMENTATION,
    TARGET_CALIBRATED_DIFFUSION_AUGMENTATION,
)
NEURAL_AUGMENTATIONS = GAN_AUGMENTATIONS + DIFFUSION_AUGMENTATIONS
GEN_AUGMENTATION_METHODS = (
    "none",
    SOURCE_GAUSSIAN_AUGMENTATION,
    TARGET_STYLE_GAUSSIAN_AUGMENTATION,
    TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
    SOURCE_GAN_AUGMENTATION,
    TARGET_STYLE_GAN_AUGMENTATION,
    TARGET_CALIBRATED_GAN_AUGMENTATION,
    SOURCE_DIFFUSION_AUGMENTATION,
    TARGET_STYLE_DIFFUSION_AUGMENTATION,
    TARGET_CALIBRATED_DIFFUSION_AUGMENTATION,
)

SOURCE_ONLY_GENERATIVE_PROTOCOL = "strict_source_only_synthetic_augmentation"
UNLABELED_TARGET_GENERATIVE_PROTOCOL = "unlabeled_target_generative_augmentation"
TARGET_CALIBRATED_GENERATIVE_PROTOCOL = "target_calibrated_generative_augmentation"


@dataclass(frozen=True, slots=True)
class GenerativeAugmentationConfig:
    """Configuration for fold-local feature-space generative augmentation."""

    method: str = "none"
    synthetic_per_class: int = 0
    noise_scale: float = 1.0
    covariance_shrinkage: float = 0.1
    covariance_floor: float = 1e-6
    random_state: int | None = 13
    target_style_strength: float = 1.0
    target_calibration_weight: float = 0.5
    neural_epochs: int = 200
    neural_hidden_dim: int = 64
    neural_batch_size: int = 32
    neural_learning_rate: float = 1e-3
    gan_latent_dim: int = 16
    gan_discriminator_steps: int = 1
    diffusion_steps: int = 24

    @property
    def enabled(self) -> bool:
        return self.method != "none" and self.synthetic_per_class > 0

    @property
    def uses_unlabeled_target_data(self) -> bool:
        return self.enabled and self.method in TARGET_STYLE_AUGMENTATIONS

    @property
    def target_calibrated(self) -> bool:
        return self.enabled and self.method in TARGET_CALIBRATED_AUGMENTATIONS

    @property
    def uses_neural_generator(self) -> bool:
        return self.enabled and self.method in NEURAL_AUGMENTATIONS

    @property
    def protocol(self) -> str:
        if self.target_calibrated:
            return TARGET_CALIBRATED_GENERATIVE_PROTOCOL
        if self.uses_unlabeled_target_data:
            return UNLABELED_TARGET_GENERATIVE_PROTOCOL
        return SOURCE_ONLY_GENERATIVE_PROTOCOL

    @property
    def protocol_category(self) -> int:
        if self.target_calibrated:
            return 3
        if self.uses_unlabeled_target_data:
            return 2
        return 1


@dataclass(frozen=True, slots=True)
class GenerativeAugmentationResult:
    """Augmented features plus provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_synthetic(self) -> int:
        return int(np.sum(self.synthetic_mask))


def generative_augmentation_config(
    *,
    method: str | None = None,
    synthetic_per_class: int | str = 0,
    noise_scale: float | str = 1.0,
    covariance_shrinkage: float | str = 0.1,
    covariance_floor: float | str = 1e-6,
    random_state: int | str | None = 13,
    target_style_strength: float | str = 1.0,
    target_calibration_weight: float | str = 0.5,
    neural_epochs: int | str = 200,
    neural_hidden_dim: int | str = 64,
    neural_batch_size: int | str = 32,
    neural_learning_rate: float | str = 1e-3,
    gan_latent_dim: int | str = 16,
    gan_discriminator_steps: int | str = 1,
    diffusion_steps: int | str = 24,
) -> GenerativeAugmentationConfig:
    """Normalize user-facing generative-augmentation options."""

    return GenerativeAugmentationConfig(
        method=normalize_generative_augmentation_method(method),
        synthetic_per_class=_normalize_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        noise_scale=_normalize_nonnegative_float(noise_scale, name="noise_scale"),
        covariance_shrinkage=_normalize_unit_interval(covariance_shrinkage, name="covariance_shrinkage"),
        covariance_floor=_normalize_nonnegative_float(covariance_floor, name="covariance_floor"),
        random_state=None if random_state in {None, "", "none", "None"} else _normalize_integer(random_state, name="random_state"),
        target_style_strength=_normalize_unit_interval(target_style_strength, name="target_style_strength"),
        target_calibration_weight=_normalize_unit_interval(target_calibration_weight, name="target_calibration_weight"),
        neural_epochs=_normalize_positive_int(neural_epochs, name="neural_epochs"),
        neural_hidden_dim=_normalize_positive_int(neural_hidden_dim, name="neural_hidden_dim"),
        neural_batch_size=_normalize_positive_int(neural_batch_size, name="neural_batch_size"),
        neural_learning_rate=_normalize_positive_float(neural_learning_rate, name="neural_learning_rate"),
        gan_latent_dim=_normalize_positive_int(gan_latent_dim, name="gan_latent_dim"),
        gan_discriminator_steps=_normalize_positive_int(gan_discriminator_steps, name="gan_discriminator_steps"),
        diffusion_steps=_normalize_positive_int(diffusion_steps, name="diffusion_steps"),
    )


def normalize_generative_augmentation_method(method: str | None) -> str:
    """Normalize aliases for feature-space generative augmentation."""

    normalized = "none" if method is None else str(method).strip().lower().replace("-", "_")
    if normalized in {"", "off", "false", "identity", "raw", "no", "disabled"}:
        normalized = "none"
    normalized = {
        "gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "class_gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "source_class_gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "source_only_gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "target_style": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "target_matched_gaussian": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "unlabeled_target_gaussian": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "style_transfer_gaussian": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "few_shot_gaussian": TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
        "calibrated_gaussian": TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
        "target_calibrated": TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
        "gan": SOURCE_GAN_AUGMENTATION,
        "source_only_gan": SOURCE_GAN_AUGMENTATION,
        "class_gan": SOURCE_GAN_AUGMENTATION,
        "target_style_neural_gan": TARGET_STYLE_GAN_AUGMENTATION,
        "unlabeled_target_gan": TARGET_STYLE_GAN_AUGMENTATION,
        "few_shot_gan": TARGET_CALIBRATED_GAN_AUGMENTATION,
        "calibrated_gan": TARGET_CALIBRATED_GAN_AUGMENTATION,
        "diffusion": SOURCE_DIFFUSION_AUGMENTATION,
        "source_only_diffusion": SOURCE_DIFFUSION_AUGMENTATION,
        "class_diffusion": SOURCE_DIFFUSION_AUGMENTATION,
        "ddpm": SOURCE_DIFFUSION_AUGMENTATION,
        "target_style_ddpm": TARGET_STYLE_DIFFUSION_AUGMENTATION,
        "target_style_neural_diffusion": TARGET_STYLE_DIFFUSION_AUGMENTATION,
        "unlabeled_target_diffusion": TARGET_STYLE_DIFFUSION_AUGMENTATION,
        "few_shot_diffusion": TARGET_CALIBRATED_DIFFUSION_AUGMENTATION,
        "calibrated_diffusion": TARGET_CALIBRATED_DIFFUSION_AUGMENTATION,
    }.get(normalized, normalized)
    if normalized not in GEN_AUGMENTATION_METHODS:
        raise ValueError(
            f"Unknown generative augmentation method {method!r}. "
            f"Available methods: {', '.join(GEN_AUGMENTATION_METHODS)}."
        )
    return normalized


def augment_training_features(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    *,
    config: GenerativeAugmentationConfig | Mapping[str, Any] | None = None,
    target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
) -> GenerativeAugmentationResult:
    """Append synthetic feature rows generated within an explicit protocol.

    ``target_labels`` are intentionally rejected.  Category-3 calibration must be
    supplied through ``target_calibration_features`` and
    ``target_calibration_labels`` so scored target labels cannot be passed
    accidentally.
    """

    cfg = _coerce_config(config)
    features = _feature_matrix(train_features, name="train_features")
    labels = _label_vector(train_labels, expected_length=features.shape[0], name="train_labels")
    if target_labels is not None:
        raise ValueError("Generative augmentation never accepts scored target_labels; use disjoint target_calibration_labels for category-3 calibration.")
    if not cfg.enabled:
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
        return GenerativeAugmentationResult(features=features, labels=labels, synthetic_mask=synthetic_mask, metadata=_metadata(cfg, features.shape[0], 0))

    target_matrix: np.ndarray | None = None
    if cfg.method in TARGET_STYLE_AUGMENTATIONS:
        if target_features is None:
            raise ValueError(f"{cfg.method} requires unlabeled target_features.")
        target_matrix = _feature_matrix(target_features, name="target_features")
        if target_matrix.shape[1] != features.shape[1]:
            raise ValueError("target_features must have the same feature dimension as train_features.")

    calibration_features: np.ndarray | None = None
    calibration_labels: np.ndarray | None = None
    if cfg.method in TARGET_CALIBRATED_AUGMENTATIONS:
        if target_calibration_features is None or target_calibration_labels is None:
            raise ValueError(f"{cfg.method} requires disjoint target_calibration_features and target_calibration_labels.")
        calibration_features = _feature_matrix(target_calibration_features, name="target_calibration_features")
        if calibration_features.shape[1] != features.shape[1]:
            raise ValueError("target_calibration_features must have the same feature dimension as train_features.")
        calibration_labels = _label_vector(
            target_calibration_labels,
            expected_length=calibration_features.shape[0],
            name="target_calibration_labels",
        )
        unknown_labels = set(np.unique(calibration_labels)) - set(np.unique(labels))
        if unknown_labels:
            raise ValueError("target_calibration_labels must be a subset of train_labels.")

    if cfg.method in GAN_AUGMENTATIONS:
        synthetic_features, synthetic_labels = _generate_neural_rows(
            features,
            labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            config=cfg,
            generator_kind="gan",
        )
    elif cfg.method in DIFFUSION_AUGMENTATIONS:
        synthetic_features, synthetic_labels = _generate_neural_rows(
            features,
            labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            config=cfg,
            generator_kind="diffusion",
        )
    else:
        synthetic_features, synthetic_labels = _generate_gaussian_rows(
            features,
            labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            config=cfg,
        )

    if synthetic_features.shape[0] == 0:
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
        return GenerativeAugmentationResult(features=features, labels=labels, synthetic_mask=synthetic_mask, metadata=_metadata(cfg, features.shape[0], 0))

    if cfg.method in TARGET_STYLE_AUGMENTATIONS and target_matrix is not None:
        synthetic_features = _match_target_style(
            synthetic_features,
            source_features=features,
            target_features=target_matrix,
            strength=cfg.target_style_strength,
            floor=cfg.covariance_floor,
        )
    if cfg.method in TARGET_CALIBRATED_AUGMENTATIONS and calibration_features is not None and calibration_labels is not None and cfg.method in NEURAL_AUGMENTATIONS:
        synthetic_features = _apply_target_calibration_shift(
            synthetic_features,
            synthetic_labels,
            source_features=features,
            source_labels=labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            weight=cfg.target_calibration_weight,
        )

    new_features = np.vstack([features, synthetic_features])
    new_labels = np.concatenate([labels, synthetic_labels])
    synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    return GenerativeAugmentationResult(
        features=new_features,
        labels=new_labels,
        synthetic_mask=synthetic_mask,
        metadata=_metadata(cfg, features.shape[0], synthetic_features.shape[0]),
    )


def make_generative_augmented_fit_model(
    fit_model: Callable[[np.ndarray, np.ndarray], Any],
    *,
    config: GenerativeAugmentationConfig | Mapping[str, Any] | None = None,
    target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray], Any]:
    """Return a ``fit_model`` wrapper that augments only the current training fold."""

    cfg = _coerce_config(config)

    def _fit(features: np.ndarray, labels: np.ndarray):
        augmented = augment_training_features(
            features,
            labels,
            config=cfg,
            target_features=target_features,
            target_calibration_features=target_calibration_features,
            target_calibration_labels=target_calibration_labels,
        )
        model = fit_model(augmented.features, augmented.labels)
        try:
            setattr(model, "generative_augmentation_metadata_", augmented.metadata)
        except Exception:  # pragma: no cover - some sklearn wrappers disallow dynamic attrs
            pass
        return model

    return _fit


def _generate_gaussian_rows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_features: np.ndarray | None,
    calibration_labels: np.ndarray | None,
    config: GenerativeAugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_state)
    synthetic_blocks: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []
    for class_label in np.unique(labels):
        class_rows = features[labels == class_label]
        if class_rows.shape[0] == 0:
            continue
        calibration_rows = None
        if calibration_features is not None and calibration_labels is not None:
            calibration_rows = calibration_features[calibration_labels == class_label]
        samples = _sample_class_rows(
            class_rows,
            fallback_rows=features,
            calibration_rows=calibration_rows,
            config=config,
            rng=rng,
        )
        synthetic_blocks.append(samples)
        synthetic_labels.append(np.full(samples.shape[0], class_label, dtype=labels.dtype))
    return _stack_synthetic_blocks(synthetic_blocks, synthetic_labels, n_features=features.shape[1], label_dtype=labels.dtype)


def _generate_neural_rows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_features: np.ndarray | None,
    calibration_labels: np.ndarray | None,
    config: GenerativeAugmentationConfig,
    generator_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch, nn, functional = _require_torch()
    _set_torch_seed(torch, config.random_state)
    training_features = features
    training_labels = labels
    if calibration_features is not None and calibration_labels is not None and calibration_features.shape[0] > 0:
        training_features = np.vstack([features, calibration_features])
        training_labels = np.concatenate([labels, calibration_labels])

    encoded_labels, class_labels = _encode_labels(training_labels)
    n_features = training_features.shape[1]
    mean = np.mean(training_features, axis=0)
    scale = np.std(training_features, axis=0)
    scale = np.where(scale <= config.covariance_floor, 1.0, scale)
    standardized = (training_features - mean) / scale

    x_train = torch.as_tensor(standardized, dtype=torch.float32)
    y_train = torch.as_tensor(encoded_labels, dtype=torch.long)
    if generator_kind == "gan":
        synthetic_standardized, synthetic_encoded = _train_and_sample_conditional_gan(
            x_train,
            y_train,
            n_classes=len(class_labels),
            n_features=n_features,
            config=config,
            torch=torch,
            nn=nn,
            functional=functional,
        )
    elif generator_kind == "diffusion":
        synthetic_standardized, synthetic_encoded = _train_and_sample_conditional_diffusion(
            x_train,
            y_train,
            n_classes=len(class_labels),
            n_features=n_features,
            config=config,
            torch=torch,
            nn=nn,
            functional=functional,
        )
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"Unknown neural generator kind: {generator_kind}")

    synthetic_features = synthetic_standardized.detach().cpu().numpy() * scale + mean
    synthetic_labels = class_labels[synthetic_encoded.detach().cpu().numpy()]
    return np.asarray(synthetic_features, dtype=float), np.asarray(synthetic_labels, dtype=labels.dtype)


def _train_and_sample_conditional_gan(
    x_train: Any,
    y_train: Any,
    *,
    n_classes: int,
    n_features: int,
    config: GenerativeAugmentationConfig,
    torch: Any,
    nn: Any,
    functional: Any,
) -> tuple[Any, Any]:
    generator = _ConditionalGenerator(
        n_features=n_features,
        n_classes=n_classes,
        latent_dim=config.gan_latent_dim,
        hidden_dim=config.neural_hidden_dim,
        nn=nn,
    )
    discriminator = _ConditionalDiscriminator(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=config.neural_hidden_dim,
        nn=nn,
    )
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.neural_learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.neural_learning_rate, betas=(0.5, 0.999))
    n_rows = x_train.shape[0]
    batch_size = min(config.neural_batch_size, n_rows)
    for _ in range(config.neural_epochs):
        permutation = torch.randperm(n_rows)
        for start in range(0, n_rows, batch_size):
            batch_index = permutation[start : start + batch_size]
            real = x_train[batch_index]
            y = y_train[batch_index]
            current_batch = real.shape[0]
            real_targets = torch.ones(current_batch, 1)
            fake_targets = torch.zeros(current_batch, 1)
            for _ in range(config.gan_discriminator_steps):
                noise = torch.randn(current_batch, config.gan_latent_dim) * config.noise_scale
                fake = generator(noise, y).detach()
                real_loss = functional.binary_cross_entropy_with_logits(discriminator(real, y), real_targets)
                fake_loss = functional.binary_cross_entropy_with_logits(discriminator(fake, y), fake_targets)
                d_loss = real_loss + fake_loss
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
            noise = torch.randn(current_batch, config.gan_latent_dim) * config.noise_scale
            fake = generator(noise, y)
            g_loss = functional.binary_cross_entropy_with_logits(discriminator(fake, y), real_targets)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
    return _sample_conditional_generator(generator, n_classes=n_classes, config=config, torch=torch)


def _sample_conditional_generator(generator: Any, *, n_classes: int, config: GenerativeAugmentationConfig, torch: Any) -> tuple[Any, Any]:
    labels = torch.arange(n_classes, dtype=torch.long).repeat_interleave(config.synthetic_per_class)
    noise = torch.randn(labels.shape[0], config.gan_latent_dim) * config.noise_scale
    with torch.no_grad():
        samples = generator(noise, labels)
    return samples, labels


def _train_and_sample_conditional_diffusion(
    x_train: Any,
    y_train: Any,
    *,
    n_classes: int,
    n_features: int,
    config: GenerativeAugmentationConfig,
    torch: Any,
    nn: Any,
    functional: Any,
) -> tuple[Any, Any]:
    model = _ConditionalDenoiser(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=config.neural_hidden_dim,
        nn=nn,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.neural_learning_rate)
    betas = torch.linspace(1e-4, 2e-2, config.diffusion_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    n_rows = x_train.shape[0]
    batch_size = min(config.neural_batch_size, n_rows)
    for _ in range(config.neural_epochs):
        permutation = torch.randperm(n_rows)
        for start in range(0, n_rows, batch_size):
            batch_index = permutation[start : start + batch_size]
            x0 = x_train[batch_index]
            y = y_train[batch_index]
            current_batch = x0.shape[0]
            t_index = torch.randint(0, config.diffusion_steps, (current_batch,), dtype=torch.long)
            noise = torch.randn_like(x0) * config.noise_scale
            alpha_bar = alpha_bars[t_index].unsqueeze(1)
            xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
            t_scaled = t_index.float().unsqueeze(1) / max(config.diffusion_steps - 1, 1)
            prediction = model(xt, y, t_scaled)
            loss = functional.mse_loss(prediction, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    labels = torch.arange(n_classes, dtype=torch.long).repeat_interleave(config.synthetic_per_class)
    samples = torch.randn(labels.shape[0], n_features) * config.noise_scale
    with torch.no_grad():
        for step in reversed(range(config.diffusion_steps)):
            t_index = torch.full((labels.shape[0], 1), step / max(config.diffusion_steps - 1, 1), dtype=torch.float32)
            predicted_noise = model(samples, labels, t_index)
            beta = betas[step]
            alpha = alphas[step]
            alpha_bar = alpha_bars[step]
            samples = (samples - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            if step > 0:
                samples = samples + torch.sqrt(beta) * torch.randn_like(samples) * config.noise_scale
    return samples, labels


class _ConditionalGenerator:  # intentionally not subclassed until torch is available
    def __new__(cls, *, n_features: int, n_classes: int, latent_dim: int, hidden_dim: int, nn: Any):  # type: ignore[override]
        class ConditionalGenerator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(n_classes, min(hidden_dim, max(n_classes, 2)))
                input_dim = latent_dim + self.embedding.embedding_dim
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, n_features),
                )

            def forward(self, noise: Any, labels: Any) -> Any:
                import torch

                return self.network(torch.cat([noise, self.embedding(labels)], dim=1))

        return ConditionalGenerator()


class _ConditionalDiscriminator:
    def __new__(cls, *, n_features: int, n_classes: int, hidden_dim: int, nn: Any):  # type: ignore[override]
        class ConditionalDiscriminator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(n_classes, min(hidden_dim, max(n_classes, 2)))
                input_dim = n_features + self.embedding.embedding_dim
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, features: Any, labels: Any) -> Any:
                import torch

                return self.network(torch.cat([features, self.embedding(labels)], dim=1))

        return ConditionalDiscriminator()


class _ConditionalDenoiser:
    def __new__(cls, *, n_features: int, n_classes: int, hidden_dim: int, nn: Any):  # type: ignore[override]
        class ConditionalDenoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(n_classes, min(hidden_dim, max(n_classes, 2)))
                input_dim = n_features + self.embedding.embedding_dim + 1
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, n_features),
                )

            def forward(self, features: Any, labels: Any, timesteps: Any) -> Any:
                import torch

                return self.network(torch.cat([features, self.embedding(labels), timesteps], dim=1))

        return ConditionalDenoiser()


def _sample_class_rows(
    class_rows: np.ndarray,
    *,
    fallback_rows: np.ndarray,
    calibration_rows: np.ndarray | None,
    config: GenerativeAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.mean(class_rows, axis=0)
    covariance_rows = class_rows
    if calibration_rows is not None and calibration_rows.shape[0] > 0:
        calibration_mean = np.mean(calibration_rows, axis=0)
        mean = (1.0 - config.target_calibration_weight) * mean + config.target_calibration_weight * calibration_mean
        covariance_rows = np.vstack([class_rows, calibration_rows])
    covariance = _regularized_covariance(
        covariance_rows,
        fallback_rows=fallback_rows,
        shrinkage=config.covariance_shrinkage,
        floor=config.covariance_floor,
    )
    sqrt_covariance = _matrix_power(covariance, 0.5, floor=config.covariance_floor)
    noise = rng.normal(size=(config.synthetic_per_class, class_rows.shape[1]))
    return mean + config.noise_scale * (noise @ sqrt_covariance.T)


def _match_target_style(
    synthetic_features: np.ndarray,
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    strength: float,
    floor: float,
) -> np.ndarray:
    if strength <= 0.0:
        return synthetic_features
    source_mean = np.mean(source_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    source_covariance = _regularized_covariance(source_features, fallback_rows=source_features, shrinkage=0.0, floor=floor)
    target_covariance = _regularized_covariance(target_features, fallback_rows=target_features, shrinkage=0.0, floor=floor)
    source_inv_sqrt = _matrix_power(source_covariance, -0.5, floor=floor)
    target_sqrt = _matrix_power(target_covariance, 0.5, floor=floor)
    matched = target_mean + (synthetic_features - source_mean) @ source_inv_sqrt @ target_sqrt
    return (1.0 - strength) * synthetic_features + strength * matched


def _apply_target_calibration_shift(
    synthetic_features: np.ndarray,
    synthetic_labels: np.ndarray,
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    weight: float,
) -> np.ndarray:
    if weight <= 0.0:
        return synthetic_features
    shifted = synthetic_features.copy()
    for class_label in np.unique(synthetic_labels):
        class_mask = synthetic_labels == class_label
        source_rows = source_features[source_labels == class_label]
        calibration_rows = calibration_features[calibration_labels == class_label]
        if source_rows.shape[0] == 0 or calibration_rows.shape[0] == 0:
            continue
        delta = np.mean(calibration_rows, axis=0) - np.mean(source_rows, axis=0)
        shifted[class_mask] = shifted[class_mask] + weight * delta
    return shifted


def _regularized_covariance(
    rows: np.ndarray,
    *,
    fallback_rows: np.ndarray,
    shrinkage: float,
    floor: float,
) -> np.ndarray:
    rows = _feature_matrix(rows, name="rows")
    fallback_rows = _feature_matrix(fallback_rows, name="fallback_rows")
    n_features = rows.shape[1]
    if rows.shape[0] > 1:
        covariance = np.cov(rows, rowvar=False)
    elif fallback_rows.shape[0] > 1:
        covariance = np.cov(fallback_rows, rowvar=False)
    else:
        covariance = np.eye(n_features, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    if not np.all(np.isfinite(covariance)):
        covariance = np.eye(n_features, dtype=float)
    return covariance + max(float(floor), 0.0) * np.eye(n_features, dtype=float)


def _matrix_power(matrix: np.ndarray, power: float, *, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    safe_values = np.maximum(values, max(float(floor), 0.0))
    return (vectors * np.power(safe_values, power)) @ vectors.T


def _coerce_config(config: GenerativeAugmentationConfig | Mapping[str, Any] | None) -> GenerativeAugmentationConfig:
    if config is None:
        return generative_augmentation_config()
    if isinstance(config, GenerativeAugmentationConfig):
        return config
    if isinstance(config, Mapping):
        return generative_augmentation_config(**dict(config))
    raise TypeError("config must be a GenerativeAugmentationConfig, a mapping, or None.")


def _metadata(config: GenerativeAugmentationConfig, n_real: int, n_synthetic: int) -> dict[str, Any]:
    return {
        "generative_augmentation_method": config.method,
        "generative_augmentation_enabled": bool(config.enabled),
        "generative_augmentation_protocol": config.protocol,
        "generative_augmentation_protocol_category": int(config.protocol_category),
        "generative_augmentation_synthetic_per_class": int(config.synthetic_per_class),
        "generative_augmentation_n_real": int(n_real),
        "generative_augmentation_n_synthetic": int(n_synthetic),
        "generative_augmentation_uses_unlabeled_target_data": bool(config.uses_unlabeled_target_data),
        "generative_augmentation_target_calibrated": bool(config.target_calibrated),
        "generative_augmentation_uses_target_labels": bool(config.target_calibrated),
        "generative_augmentation_uses_neural_generator": bool(config.uses_neural_generator),
        "generative_augmentation_valid_for_strict_source_only": bool(config.enabled and config.protocol_category == 1),
        "generative_augmentation_valid_for_benchmark": bool(config.protocol_category in {1, 2, 3}),
        "generative_augmentation_protocol_note": _protocol_note(config),
    }


def _protocol_note(config: GenerativeAugmentationConfig) -> str:
    if not config.enabled:
        return ""
    if config.method in SOURCE_ONLY_AUGMENTATIONS:
        return "source-only synthetic feature augmentation"
    if config.method in TARGET_STYLE_AUGMENTATIONS:
        return "uses unlabeled target features for distribution/style matching; category-2 target-adaptive augmentation"
    return "uses disjoint labeled target calibration rows; category-3 supervised/calibrated augmentation"


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return matrix


def _label_vector(labels: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector


def _stack_synthetic_blocks(
    synthetic_blocks: list[np.ndarray],
    synthetic_labels: list[np.ndarray],
    *,
    n_features: int,
    label_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if not synthetic_blocks:
        return np.empty((0, n_features), dtype=float), np.asarray([], dtype=label_dtype)
    return np.vstack(synthetic_blocks), np.concatenate(synthetic_labels)


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_labels = np.unique(labels)
    lookup = {label: index for index, label in enumerate(class_labels)}
    encoded = np.asarray([lookup[label] for label in labels], dtype=int)
    return encoded, class_labels


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise ImportError(
            "Neural generative augmentation methods require the optional torch extra. "
            "Install NeuRepTrace with the 'torch' or 'ml' extra, or use a Gaussian augmentation method."
        ) from exc
    return torch, nn, functional


def _set_torch_seed(torch: Any, random_state: int | None) -> None:
    if random_state is None:
        return
    torch.manual_seed(int(random_state))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch versions
        pass


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


def _normalize_positive_int(value: int | str, *, name: str) -> int:
    integer = _normalize_integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive.")
    return integer


def _normalize_nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


def _normalize_positive_float(value: float | str, *, name: str) -> float:
    numeric = _normalize_nonnegative_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _normalize_unit_interval(value: float | str, *, name: str) -> float:
    numeric = _normalize_nonnegative_float(value, name=name)
    if numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric
