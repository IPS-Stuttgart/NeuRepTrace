"""MEG-specific reusable utilities for NeuRepTrace.

This subpackage contains generic sensor-level MEG helpers that are useful beyond
one dataset: FieldTrip-like raw/trial struct accessors, alpha-band signal
features, sensor geometry projection, per-trial alpha metrics, and alpha-power
centroid trajectories.
"""

from neureptrace.meg.alpha_metrics import AlphaMetricConfig, compute_alpha_metrics, compute_alpha_trial_metrics
from neureptrace.meg.alpha_movement import AlphaMovementConfig, compute_alpha_movement, compute_alpha_movement_trajectory, summarize_alpha_movement
from neureptrace.meg.alpha_signal import extract_alpha_signal_and_phase, extract_phase, extract_time_basis

__all__ = [
    "AlphaMetricConfig",
    "AlphaMovementConfig",
    "compute_alpha_metrics",
    "compute_alpha_movement",
    "compute_alpha_movement_trajectory",
    "compute_alpha_trial_metrics",
    "extract_alpha_signal_and_phase",
    "extract_phase",
    "extract_time_basis",
    "summarize_alpha_movement",
]
