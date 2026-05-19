"""Synthetic M/EEG-like fixtures for NeuRepTrace tests and demos.

The helpers in this module generate private-data-free epoch arrays with a
balanced class structure, a localized evoked class signal, optional
session/group labels, and optional participant-specific sensor transforms.
They are intended for smoke tests of decoding, calibration, windowing,
cross-subject alignment, and leakage controls without depending on PyMEGDec's
MATLAB file conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticEpochConfig:
    """Configuration for a balanced synthetic M/EEG epoch set.

    Generated data are returned as ``trials x channels x time`` arrays.
    Labels are balanced across ``repeats_per_class`` groups; each group
    contains one trial for every class. This makes the fixture useful for
    grouped cross-validation tests as well as simple feature-matrix tests.
    """

    n_classes: int = 4
    repeats_per_class: int = 8
    n_channels: int = 12
    n_times: int = 101
    tmin: float = -0.2
    tmax: float = 0.6
    signal_window: tuple[float, float] = (0.12, 0.22)
    signal_scale: float = 3.0
    noise_scale: float = 0.2
    oscillation_scale: float = 0.02
    group_shift_scale: float = 0.0
    label_start: int = 0
    random_seed: int = 13
    shuffle_trials: bool = True


@dataclass(frozen=True)
class SyntheticEpochs:
    """In-memory synthetic epochs and decoding metadata."""

    data: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    groups: np.ndarray
    channel_names: tuple[str, ...]
    sensor_positions: np.ndarray
    participant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        """Number of generated trials."""

        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of generated channels."""

        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        """Number of generated time samples."""

        return int(self.data.shape[2])

    @property
    def n_classes(self) -> int:
        """Number of distinct labels."""

        return int(np.unique(self.labels).size)


def make_synthetic_epochs(
    config: SyntheticEpochConfig | None = None,
    *,
    class_prototypes: np.ndarray | None = None,
    sensor_transform: np.ndarray | None = None,
    participant_id: str | None = None,
    rng: np.random.Generator | None = None,
) -> SyntheticEpochs:
    """Return a deterministic synthetic epoch set.

    Parameters
    ----------
    config:
        Synthetic data parameters. Defaults are chosen to create a small,
        strongly decodable fixture that still contains noise and a weak
        alpha-like carrier.
    class_prototypes:
        Optional ``n_classes x n_channels`` prototype matrix. Supplying this
        allows several participants to share latent class structure.
    sensor_transform:
        Optional ``n_channels x n_channels`` linear transform applied to class
        prototypes before generating the participant's epochs.
    participant_id:
        Optional participant label stored in the returned object.
    rng:
        Optional generator for callers that need independent streams while
        preserving shared prototypes.
    """

    config = config or SyntheticEpochConfig()
    _validate_config(config)
    rng = rng or np.random.default_rng(config.random_seed)

    times = np.linspace(config.tmin, config.tmax, config.n_times)
    labels, groups = _balanced_labels_and_groups(config)

    prototypes = _prototype_matrix(class_prototypes, config, rng)
    if sensor_transform is not None:
        transform = _sensor_transform_matrix(sensor_transform, config.n_channels)
        prototypes = prototypes @ transform.T
        prototypes = _normalize_rows(prototypes)

    envelope = _signal_envelope(times, config.signal_window)
    data = rng.normal(
        scale=config.noise_scale,
        size=(labels.size, config.n_channels, config.n_times),
    )

    if config.oscillation_scale:
        data += _oscillation(times, config.n_channels, config.oscillation_scale)[None, :, :]

    if config.group_shift_scale:
        shifts = rng.normal(
            scale=config.group_shift_scale,
            size=(config.repeats_per_class, config.n_channels, 1),
        )
        data += shifts[groups]

    for trial_index, label in enumerate(labels):
        class_index = int(label) - config.label_start
        data[trial_index] += config.signal_scale * prototypes[class_index][:, None] * envelope[None, :]
        data[trial_index] += 1e-5 * (trial_index + 1)

    if config.shuffle_trials:
        order = rng.permutation(labels.size)
        data = data[order]
        labels = labels[order]
        groups = groups[order]

    metadata = {
        "config": asdict(config),
        "signal_window": tuple(config.signal_window),
        "participant_id": participant_id,
    }
    return SyntheticEpochs(
        data=data,
        labels=labels,
        times=times,
        groups=groups,
        channel_names=_channel_names(config.n_channels),
        sensor_positions=_channel_positions(config.n_channels),
        participant_id=participant_id,
        metadata=metadata,
    )


def make_synthetic_participant_epochs(
    n_participants: int,
    config: SyntheticEpochConfig | None = None,
    *,
    transform_strength: float = 0.25,
    participant_shift_scale: float = 0.0,
    random_seed: int | None = None,
) -> list[SyntheticEpochs]:
    """Return participants with shared labels and transformed class structure.

    The participants share one prototype matrix, but each participant receives
    an independent noise stream and a deterministic sensor-space transform.
    This creates a compact fixture for cross-subject decoding and alignment
    tests without relying on a private MEG dataset.
    """

    if n_participants < 1:
        raise ValueError("n_participants must be at least 1.")
    if not 0.0 <= transform_strength <= 1.0:
        raise ValueError("transform_strength must be between 0 and 1.")
    if participant_shift_scale < 0.0:
        raise ValueError("participant_shift_scale must be non-negative.")

    config = config or SyntheticEpochConfig()
    _validate_config(config)
    seed = config.random_seed if random_seed is None else int(random_seed)
    seed_sequence = np.random.SeedSequence(seed)
    prototype_seed, order_seed, *participant_seeds = seed_sequence.spawn(n_participants + 2)
    prototype_rng = np.random.default_rng(prototype_seed)
    prototypes = _class_prototypes(prototype_rng, config.n_classes, config.n_channels)

    shared_order = None
    if config.shuffle_trials:
        order_rng = np.random.default_rng(order_seed)
        shared_order = order_rng.permutation(config.n_classes * config.repeats_per_class)

    participants: list[SyntheticEpochs] = []
    for index, participant_seed in enumerate(participant_seeds, start=1):
        participant_rng = np.random.default_rng(participant_seed)
        participant_config = replace(
            config,
            random_seed=int(participant_seed.generate_state(1)[0]),
            shuffle_trials=False,
        )
        transform = _participant_transform(
            participant_rng,
            config.n_channels,
            strength=transform_strength,
        )
        participant_id = f"sub-{index:02d}"
        epochs = make_synthetic_epochs(
            participant_config,
            class_prototypes=prototypes,
            sensor_transform=transform,
            participant_id=participant_id,
            rng=participant_rng,
        )
        if shared_order is not None:
            epochs = SyntheticEpochs(
                data=epochs.data[shared_order],
                labels=epochs.labels[shared_order],
                times=epochs.times,
                groups=epochs.groups[shared_order],
                channel_names=epochs.channel_names,
                sensor_positions=epochs.sensor_positions,
                participant_id=epochs.participant_id,
                metadata={
                    **epochs.metadata,
                    "config": asdict(config),
                    "shared_trial_order": True,
                },
            )
        if participant_shift_scale:
            shift = participant_rng.normal(
                scale=participant_shift_scale,
                size=(1, config.n_channels, 1),
            )
            epochs = SyntheticEpochs(
                data=epochs.data + shift,
                labels=epochs.labels,
                times=epochs.times,
                groups=epochs.groups,
                channel_names=epochs.channel_names,
                sensor_positions=epochs.sensor_positions,
                participant_id=epochs.participant_id,
                metadata={
                    **epochs.metadata,
                    "participant_shift_scale": participant_shift_scale,
                },
            )
        participants.append(epochs)

    return participants


def window_feature_matrix(
    epochs: SyntheticEpochs,
    window: tuple[float, float] | None = None,
    *,
    reducer: str = "mean",
) -> np.ndarray:
    """Convert synthetic epochs to a decoding-ready feature matrix.

    Parameters
    ----------
    epochs:
        Synthetic epoch object returned by :func:`make_synthetic_epochs`.
    window:
        Inclusive time interval. ``None`` uses the full epoch.
    reducer:
        ``"mean"`` averages over selected time samples and returns
        ``trials x channels`` features. ``"flatten"`` returns
        ``trials x (channels * selected_times)`` features.
    """

    data = np.asarray(epochs.data, dtype=float)
    if data.ndim != 3:
        raise ValueError("epochs.data must be a trials x channels x time array.")

    if window is None:
        selected = data
    else:
        start, stop = window
        if start >= stop:
            raise ValueError("window start must be smaller than stop.")
        time_mask = (epochs.times >= start) & (epochs.times <= stop)
        if not np.any(time_mask):
            raise ValueError("window must overlap epochs.times.")
        selected = data[:, :, time_mask]

    if reducer == "mean":
        return selected.mean(axis=2)
    if reducer == "flatten":
        return selected.reshape(selected.shape[0], -1)
    raise ValueError("reducer must be 'mean' or 'flatten'.")


def _validate_config(config: SyntheticEpochConfig) -> None:
    positive_integer_fields = (
        ("n_classes", 2, "n_classes must be at least 2."),
        ("repeats_per_class", 1, "repeats_per_class must be at least 1."),
        ("n_channels", 1, "n_channels must be at least 1."),
        ("n_times", 3, "n_times must be at least 3."),
    )
    for field_name, minimum, message in positive_integer_fields:
        if int(getattr(config, field_name)) < minimum:
            raise ValueError(message)

    start, stop = config.signal_window
    if config.tmin >= config.tmax:
        raise ValueError("tmin must be smaller than tmax.")
    if start >= stop:
        raise ValueError("signal_window start must be smaller than stop.")
    if stop < config.tmin or start > config.tmax:
        raise ValueError("signal_window must overlap the generated time vector.")

    non_negative_fields = (
        "noise_scale",
        "oscillation_scale",
        "group_shift_scale",
    )
    if config.signal_scale <= 0.0:
        raise ValueError("signal_scale must be positive.")
    for field_name in non_negative_fields:
        if getattr(config, field_name) < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")


def _balanced_labels_and_groups(config: SyntheticEpochConfig) -> tuple[np.ndarray, np.ndarray]:
    class_labels = np.arange(
        config.label_start,
        config.label_start + config.n_classes,
        dtype=int,
    )
    labels = np.tile(class_labels, config.repeats_per_class)
    groups = np.repeat(np.arange(config.repeats_per_class, dtype=int), config.n_classes)
    return labels, groups


def _prototype_matrix(
    class_prototypes: np.ndarray | None,
    config: SyntheticEpochConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if class_prototypes is None:
        return _class_prototypes(rng, config.n_classes, config.n_channels)
    prototypes = np.asarray(class_prototypes, dtype=float)
    expected = (config.n_classes, config.n_channels)
    if prototypes.shape != expected:
        raise ValueError(f"class_prototypes must have shape {expected}, got {prototypes.shape}.")
    return _normalize_rows(prototypes)


def _class_prototypes(
    rng: np.random.Generator,
    n_classes: int,
    n_channels: int,
) -> np.ndarray:
    prototypes = rng.normal(size=(n_classes, n_channels))
    return _normalize_rows(prototypes)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(float).eps)


def _signal_envelope(times: np.ndarray, signal_window: tuple[float, float]) -> np.ndarray:
    start, stop = signal_window
    center = 0.5 * (start + stop)
    width = max((stop - start) / 4.0, np.finfo(float).eps)
    envelope = np.exp(-0.5 * ((times - center) / width) ** 2)
    envelope[(times < start) | (times > stop)] = 0.0
    return envelope


def _oscillation(times: np.ndarray, n_channels: int, scale: float) -> np.ndarray:
    phase = np.linspace(0.0, np.pi, n_channels, endpoint=False)[:, None]
    return scale * np.sin(2.0 * np.pi * 10.0 * times[None, :] + phase)


def _channel_names(n_channels: int) -> tuple[str, ...]:
    return tuple(f"MEG{index + 1:03d}" for index in range(n_channels))


def _channel_positions(n_channels: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False)
    radius = 0.08
    return np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            0.02 * np.sin(2.0 * angles),
        ]
    )


def _sensor_transform_matrix(
    sensor_transform: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    transform = np.asarray(sensor_transform, dtype=float)
    expected = (n_channels, n_channels)
    if transform.shape != expected:
        raise ValueError(f"sensor_transform must have shape {expected}, got {transform.shape}.")
    return transform


def _participant_transform(
    rng: np.random.Generator,
    n_channels: int,
    *,
    strength: float,
) -> np.ndarray:
    if strength == 0.0:
        return np.eye(n_channels)
    random_matrix = rng.normal(size=(n_channels, n_channels))
    q_matrix, _ = np.linalg.qr(random_matrix)
    if np.linalg.det(q_matrix) < 0.0:
        q_matrix[:, 0] *= -1.0
    transform = (1.0 - strength) * np.eye(n_channels) + strength * q_matrix
    column_norms = np.linalg.norm(transform, axis=0, keepdims=True)
    return transform / np.maximum(column_norms, np.finfo(float).eps)
