"""Patch pseudo-label target-alignment fallback validity metadata.

Pseudo-label target-calibrated alignment is a category-2/transductive protocol: it
uses held-out target features and classifier-generated pseudo labels while fitting
or attempting to fit the target projection.  If a fold falls back to the source
only group projection because pseudo-label alignment cannot be fit, the row still
belongs to that requested protocol and must not be marked as benchmark-valid or
strict source-only.  This patch keeps the existing fallback behavior but corrects
its metadata and diagnostics.
"""

from __future__ import annotations

from dataclasses import replace

_PATCH_MARKER = "_neureptrace_mne_pseudo_alignment_fallback_validity_patch_installed"


def install() -> None:
    """Install pseudo-label fallback validity metadata corrections."""

    from neureptrace import mne_time_decode

    if getattr(mne_time_decode, _PATCH_MARKER, False):
        return

    original_fallback = mne_time_decode._pseudo_label_target_alignment_fallback_result

    def _pseudo_label_target_alignment_fallback_result(result, alignment_config, *, stop_reason: str):
        fallback = original_fallback(result, alignment_config, stop_reason=stop_reason)
        protocol_metadata = dict(alignment_config.static_metadata())
        forced = {
            "alignment_target_projection": mne_time_decode.PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
            "alignment_target_projection_fit": "source_group_projection_fallback",
            "alignment_target_alignment_rows": 0,
            "alignment_target_labels_used": False,
            "alignment_target_pseudo_labels_used": False,
            "alignment_target_anchor_values_used": False,
            "alignment_pseudo_label_target_calibrated": True,
            "alignment_pseudo_label_fallback": True,
            "alignment_pseudo_label_fallback_reason": stop_reason,
            "alignment_valid_for_benchmark": False,
            "alignment_valid_for_strict_source_only": False,
        }
        metadata = {
            **fallback.metadata,
            **protocol_metadata,
            **forced,
        }
        diagnostics = {
            **fallback.diagnostics,
            **protocol_metadata,
            "uses_unlabeled_target_data": True,
            "target_transform_type": "source_group_projection_fallback",
            **forced,
        }
        return replace(fallback, metadata=metadata, diagnostics=diagnostics)

    mne_time_decode._pseudo_label_target_alignment_fallback_result = _pseudo_label_target_alignment_fallback_result
    setattr(mne_time_decode, _PATCH_MARKER, True)
