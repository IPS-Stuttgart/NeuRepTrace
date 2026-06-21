"""
Strict source-only spatiotemporal fusion CNN for MEG LOSO classification.

Input convention:
    X: numpy array or torch tensor with shape [n_trials, n_channels, n_times]
    y: integer labels with shape [n_trials]
    subjects: subject identifiers with shape [n_trials]

The provided LOSO functions deliberately fit the channel standardizer and choose the
early-stopping epoch using source subjects only. The held-out subject is never used
for fitting, validation, scaling, feature selection, or early stopping.
"""

from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class Window:
    """A temporal window in sample indices: x[:, :, start:stop]."""

    name: str
    start: int
    stop: int


def seed_everything(seed: int) -> None:
    """Set random seeds for repeatable folds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_key(name: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    if not key:
        key = "branch"
    if key[0].isdigit():
        key = "b_" + key
    return key


def windows_from_ms(
    windows_ms: Sequence[Tuple[str, float, float]],
    *,
    sfreq: float,
    tmin_ms: float,
    n_times: int,
) -> List[Window]:
    """
    Convert windows from milliseconds relative to event onset into sample-index windows.

    Args:
        windows_ms:
            Sequence such as [("early", 80, 150), ("mid", 150, 300)].
        sfreq:
            Sampling rate in Hz.
        tmin_ms:
            Time of X[:, :, 0] in milliseconds relative to event onset.
            For an epoch from -200 ms to 800 ms, use tmin_ms=-200.
        n_times:
            Number of time samples in X.

    Returns:
        List of Window objects clipped to [0, n_times].
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")

    windows: List[Window] = []
    for name, start_ms, stop_ms in windows_ms:
        start = int(round((start_ms - tmin_ms) * sfreq / 1000.0))
        stop = int(round((stop_ms - tmin_ms) * sfreq / 1000.0))
        start = max(0, min(start, n_times))
        stop = max(0, min(stop, n_times))
        if stop <= start:
            raise ValueError(
                f"Invalid/clipped window {name!r}: start={start}, stop={stop}. "
                "Check sfreq, tmin_ms, and n_times."
            )
        windows.append(Window(name=name, start=start, stop=stop))
    return windows


def default_three_windows(n_times: int) -> List[Window]:
    """Fallback when no physiological timing is supplied."""
    a = n_times // 3
    b = 2 * n_times // 3
    return [
        Window("early_third", 0, a),
        Window("middle_third", a, b),
        Window("late_third", b, n_times),
    ]


def baseline_correct_per_trial(
    X: np.ndarray,
    *,
    baseline: Tuple[int, int],
    copy_array: bool = True,
) -> np.ndarray:
    """
    Per-trial baseline correction, which is normally safe for strict LOSO because
    each trial is corrected using only its own baseline samples.

    Args:
        X: [trials, channels, time].
        baseline: sample-index interval [start, stop) used as baseline.
        copy_array: if True, do not modify X in place.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape [trials, channels, time].")
    start, stop = baseline
    if not (0 <= start < stop <= X.shape[-1]):
        raise ValueError(f"Invalid baseline slice {baseline} for n_times={X.shape[-1]}.")
    out = X.copy() if copy_array else X
    out -= out[:, :, start:stop].mean(axis=2, keepdims=True)
    return out


class SourceOnlyStandardizer:
    """
    Channelwise z-score standardizer fit on source-training trials only.

    This computes one mean/std per channel over all training trials and time samples:
        mean shape: [1, channels, 1]
        std shape:  [1, channels, 1]

    Do not fit this on held-out-subject data if you want strict no-calibration LOSO.
    """

    def __init__(self, eps: float = 1e-6, with_mean: bool = True, with_std: bool = True):
        self.eps = eps
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SourceOnlyStandardizer":
        if X.ndim != 3:
            raise ValueError("X must have shape [trials, channels, time].")
        if self.with_mean:
            self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        else:
            self.mean_ = np.zeros((1, X.shape[1], 1), dtype=np.float32)

        if self.with_std:
            self.std_ = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
            self.std_ = np.maximum(self.std_, self.eps)
        else:
            self.std_ = np.ones((1, X.shape[1], 1), dtype=np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call fit before transform.")
        return ((X.astype(np.float32, copy=False) - self.mean_) / self.std_).astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MEGArrayDataset(Dataset):
    """Simple array-backed dataset for MEG classification."""

    def __init__(self, X: ArrayLike, y: ArrayLike):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32, copy=False))
        else:
            X = X.float()

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y.astype(np.int64, copy=False))
        else:
            y = y.long()

        if X.ndim != 3:
            raise ValueError("X must have shape [trials, channels, time].")
        if y.ndim != 1:
            raise ValueError("y must have shape [trials].")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of trials.")

        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class EEGNetBranch(nn.Module):
    """
    Compact EEGNet-like branch for one temporal window or region.

    Input:  [batch, channels, time]
    Output: [batch, f2]

    Structure:
        temporal convolution
        depthwise spatial convolution across channels
        separable temporal convolution
        adaptive pooling to one vector
    """

    def __init__(
        self,
        n_chans: int,
        *,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 31,
        separable_kernel: int = 15,
        pool: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        if n_chans <= 0:
            raise ValueError("n_chans must be positive.")
        if f1 <= 0 or depth_multiplier <= 0 or f2 <= 0:
            raise ValueError("f1, depth_multiplier, and f2 must be positive.")
        if temporal_kernel % 2 == 0 or separable_kernel % 2 == 0:
            raise ValueError("Use odd temporal_kernel and separable_kernel for same-length padding.")
        if pool <= 0:
            raise ValueError("pool must be positive.")

        f1d = f1 * depth_multiplier
        self.out_features = f2

        self.temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=f1,
                out_channels=f1d,
                kernel_size=(n_chans, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool), ceil_mode=True),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                in_channels=f1d,
                out_channels=f1d,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f1d,
                bias=False,
            ),
            nn.Conv2d(in_channels=f1d, out_channels=f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("EEGNetBranch expects [batch, channels, time].")
        x = x.unsqueeze(1)  # [batch, 1, channels, time]
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x.flatten(1)


class SpatioTemporalFusionCNN(nn.Module):
    """
    Parallel-window MEG CNN for strict LOSO visual classification.

    The model concatenates:
        1. a full-epoch EEGNet-like branch,
        2. separate EEGNet-like branches for early/mid/late windows,
        3. optional sensor-region branches.

    This is intended to test the "spatiotemporal fusion" idea:
    specialized branches can focus on different visual-processing latencies,
    while the full branch can still learn broader dynamics.
    """

    def __init__(
        self,
        *,
        n_chans: int,
        n_times: int,
        n_classes: int,
        windows: Optional[Sequence[Window]] = None,
        region_indices: Optional[Mapping[str, Sequence[int]]] = None,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 31,
        separable_kernel: int = 15,
        pool: int = 4,
        dropout: float = 0.25,
        hidden: int = 128,
    ):
        super().__init__()
        if n_chans <= 0 or n_times <= 0 or n_classes <= 1:
            raise ValueError("n_chans and n_times must be positive; n_classes must be > 1.")
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_classes = n_classes

        if windows is None:
            windows = default_three_windows(n_times)

        cleaned_windows: List[Tuple[str, Window]] = []
        seen_keys: set[str] = set()
        for i, w in enumerate(windows):
            if not (0 <= w.start < w.stop <= n_times):
                raise ValueError(f"Invalid window {w}: must satisfy 0 <= start < stop <= n_times.")
            key = _safe_key(w.name)
            if key in seen_keys:
                key = f"{key}_{i}"
            seen_keys.add(key)
            cleaned_windows.append((key, w))
        self.window_specs = cleaned_windows

        self.full_branch = EEGNetBranch(
            n_chans,
            f1=f1,
            depth_multiplier=depth_multiplier,
            f2=f2,
            temporal_kernel=temporal_kernel,
            separable_kernel=separable_kernel,
            pool=pool,
            dropout=dropout,
        )

        self.window_branches = nn.ModuleDict(
            {
                key: EEGNetBranch(
                    n_chans,
                    f1=f1,
                    depth_multiplier=depth_multiplier,
                    f2=f2,
                    temporal_kernel=temporal_kernel,
                    separable_kernel=separable_kernel,
                    pool=pool,
                    dropout=dropout,
                )
                for key, _ in cleaned_windows
            }
        )

        self.region_indices: Dict[str, torch.Tensor] = {}
        self.region_branches = nn.ModuleDict()
        if region_indices:
            seen_region_keys: set[str] = set()
            for i, (name, idxs) in enumerate(region_indices.items()):
                idx_array = np.asarray(list(idxs), dtype=np.int64)
                if idx_array.ndim != 1 or len(idx_array) == 0:
                    raise ValueError(f"Region {name!r} must contain at least one channel index.")
                if idx_array.min() < 0 or idx_array.max() >= n_chans:
                    raise ValueError(f"Region {name!r} has indices outside [0, {n_chans}).")
                key = _safe_key(name)
                if key in seen_region_keys:
                    key = f"{key}_{i}"
                seen_region_keys.add(key)
                self.region_indices[key] = torch.as_tensor(idx_array, dtype=torch.long)
                self.region_branches[key] = EEGNetBranch(
                    len(idx_array),
                    f1=f1,
                    depth_multiplier=depth_multiplier,
                    f2=f2,
                    temporal_kernel=temporal_kernel,
                    separable_kernel=separable_kernel,
                    pool=pool,
                    dropout=dropout,
                )

        n_branches = 1 + len(self.window_branches) + len(self.region_branches)
        fusion_dim = n_branches * f2

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("SpatioTemporalFusionCNN expects [batch, channels, time].")
        if x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(f"Expected [batch, {self.n_chans}, {self.n_times}], got {tuple(x.shape)}.")

        features: List[torch.Tensor] = [self.full_branch(x)]

        for key, window in self.window_specs:
            xw = x[:, :, window.start : window.stop]
            features.append(self.window_branches[key](xw))

        for key, branch in self.region_branches.items():
            idx = self.region_indices[key].to(device=x.device)
            xr = x.index_select(dim=1, index=idx)
            features.append(branch(xr))

        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        z = self.extract_features(x)
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits


def make_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Balanced inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(y_train) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: Union[str, torch.device],
    n_classes: int,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Return loss, accuracy, and balanced accuracy."""
    model.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    losses: List[float] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        if criterion is not None:
            losses.append(float(criterion(logits, yb).detach().cpu()))
        pred = logits.argmax(dim=1)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    accuracy = float((yt == yp).mean())

    per_class = []
    for c in range(n_classes):
        mask = yt == c
        if mask.any():
            per_class.append(float((yp[mask] == yt[mask]).mean()))
    balanced_accuracy = float(np.mean(per_class)) if per_class else math.nan

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "n": int(len(yt)),
    }


def _source_subject_train_val_masks(
    subjects: np.ndarray,
    *,
    heldout_subject,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train, validation, test masks.

    Test is the held-out subject.
    Validation is a subset of source subjects only.
    """
    subjects = np.asarray(subjects)
    test_mask = subjects == heldout_subject
    source_mask = ~test_mask

    source_subjects = np.unique(subjects[source_mask])
    if len(source_subjects) < 2:
        raise ValueError("Need at least two source subjects to create a source-only validation split.")

    rng = np.random.default_rng(seed)
    shuffled = source_subjects.copy()
    rng.shuffle(shuffled)
    n_val = int(round(val_frac * len(shuffled)))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_subjects = set(shuffled[:n_val].tolist())

    val_mask = source_mask & np.isin(subjects, list(val_subjects))
    train_mask = source_mask & ~val_mask
    return train_mask, val_mask, test_mask


def train_one_strict_loso_fold(
    *,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    heldout_subject,
    model_factory: Callable[[], nn.Module],
    n_classes: int,
    batch_size: int = 64,
    max_epochs: int = 150,
    patience: int = 25,
    val_frac: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_class_weights: bool = True,
    standardize: bool = True,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    num_workers: int = 0,
) -> Tuple[nn.Module, SourceOnlyStandardizer, Dict[str, Any]]:
    """
    Train/evaluate one held-out-subject fold without target calibration.

    The held-out subject is used only once: final evaluation.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape [trials, channels, time].")
    y = np.asarray(y).astype(np.int64)
    subjects = np.asarray(subjects)
    if len(X) != len(y) or len(y) != len(subjects):
        raise ValueError("X, y, and subjects must have matching first dimensions.")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be between 0 and 1.")

    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    train_mask, val_mask, test_mask = _source_subject_train_val_masks(subjects, heldout_subject=heldout_subject, val_frac=val_frac, seed=seed)

    scaler = SourceOnlyStandardizer()
    if standardize:
        X_train = scaler.fit_transform(X[train_mask])
        X_val = scaler.transform(X[val_mask])
        X_test = scaler.transform(X[test_mask])
    else:
        # Fit a do-nothing scaler for consistent return type.
        scaler.fit(np.zeros((1, X.shape[1], X.shape[2]), dtype=np.float32))
        scaler.mean_[:] = 0
        scaler.std_[:] = 1
        X_train = X[train_mask].astype(np.float32, copy=False)
        X_val = X[val_mask].astype(np.float32, copy=False)
        X_test = X[test_mask].astype(np.float32, copy=False)

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        MEGArrayDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        MEGArrayDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        MEGArrayDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model_factory().to(device)

    if use_class_weights:
        weights = make_class_weights(y_train, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_val_bal_acc = -math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_metrics = evaluate_classifier(model, val_loader, device=device, n_classes=n_classes, criterion=criterion)

        # Main early-stopping key: source-subject balanced accuracy.
        improved = val_metrics["balanced_accuracy"] > best_val_bal_acc or (
            math.isclose(val_metrics["balanced_accuracy"], best_val_bal_acc) and val_metrics["loss"] < best_val_loss
        )
        if improved:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, device=device, n_classes=n_classes, criterion=criterion)

    metrics: Dict[str, Any] = {
        "heldout_subject": heldout_subject,
        "best_epoch": float(best_epoch),
        "best_source_val_balanced_accuracy": float(best_val_bal_acc),
        "best_source_val_loss": float(best_val_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "n_train_trials": int(train_mask.sum()),
        "n_val_trials": int(val_mask.sum()),
        "n_test_trials": int(test_mask.sum()),
        "n_train_subjects": int(len(np.unique(subjects[train_mask]))),
        "n_val_subjects": int(len(np.unique(subjects[val_mask]))),
    }
    return model, scaler, metrics


def run_strict_loso(
    *,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_factory: Callable[[], nn.Module],
    n_classes: int,
    seed: int = 0,
    **train_kwargs,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run strict leave-one-subject-out evaluation.

    Args:
        X, y, subjects:
            Full dataset. X has shape [trials, channels, time].
        model_factory:
            Zero-argument callable returning a fresh model for each fold.
        n_classes:
            Number of labels/classes.
        seed:
            Base seed. Fold i uses seed+i.
        **train_kwargs:
            Forwarded to train_one_strict_loso_fold.

    Returns:
        fold_results, summary
    """
    subjects = np.asarray(subjects)
    unique_subjects = np.unique(subjects)
    fold_results: List[Dict[str, Any]] = []

    for i, heldout in enumerate(unique_subjects):
        _, _, metrics = train_one_strict_loso_fold(
            X=X,
            y=y,
            subjects=subjects,
            heldout_subject=heldout,
            model_factory=model_factory,
            n_classes=n_classes,
            seed=seed + i,
            **train_kwargs,
        )
        fold_results.append(metrics)

    accs = np.array([m["test_accuracy"] for m in fold_results], dtype=float)
    baccs = np.array([m["test_balanced_accuracy"] for m in fold_results], dtype=float)

    summary = {
        "n_folds": int(len(fold_results)),
        "mean_test_accuracy": float(np.mean(accs)),
        "std_test_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_test_balanced_accuracy": float(np.mean(baccs)),
        "std_test_balanced_accuracy": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
        "chance_accuracy": float(1.0 / n_classes),
        "chance_ratio_accuracy": float(np.mean(accs) / (1.0 / n_classes)),
    }
    return fold_results, summary


# Example configuration:
#
# windows = windows_from_ms(
#     [
#         ("early_visual_80_150ms", 80, 150),
#         ("mid_visual_150_300ms", 150, 300),
#         ("late_visual_300_600ms", 300, 600),
#     ],
#     sfreq=1000.0,
#     tmin_ms=-200.0,
#     n_times=X.shape[2],
# )
#
# model_factory = lambda: SpatioTemporalFusionCNN(
#     n_chans=X.shape[1],
#     n_times=X.shape[2],
#     n_classes=16,
#     windows=windows,
#     # Optional: pass channel groups if you have channel indices.
#     # region_indices={"posterior": posterior_idx, "occipital": occipital_idx},
#     f1=8,
#     depth_multiplier=2,
#     f2=16,
#     hidden=128,
#     dropout=0.35,
# )
#
# fold_results, summary = run_strict_loso(
#     X=X,
#     y=y,
#     subjects=subjects,
#     model_factory=model_factory,
#     n_classes=16,
#     batch_size=64,
#     max_epochs=150,
#     patience=25,
#     lr=1e-3,
#     weight_decay=1e-4,
#     val_frac=0.2,
#     use_class_weights=True,
#     standardize=True,
# )
# print(summary)
