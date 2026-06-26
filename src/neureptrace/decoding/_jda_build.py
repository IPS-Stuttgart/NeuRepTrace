from __future__ import annotations

import numpy as np

from neureptrace.decoding._jda_types import JDA_CATEGORY, JDA_PROTOCOL, JointDistributionResult


def build_result(prepared, solved, *, conditional_weight, balance_source_classes):
    labels = np.empty(prepared.target.shape[0], dtype=object)
    for row, class_index in enumerate(solved.assignments.tolist()):
        labels[row] = prepared.classes[class_index]
    metadata = {
        "jda_protocol": JDA_PROTOCOL,
        "jda_category": JDA_CATEGORY,
        "jda_uses_source_labels": True,
        "jda_uses_unlabeled_target_features": True,
        "jda_uses_target_labels": False,
        "jda_valid_for_source_only": False,
        "jda_valid_for_unlabeled_adaptation": True,
        "jda_iterations": int(solved.iterations),
        "jda_converged": bool(solved.converged),
        "jda_components": int(prepared.components),
        "jda_conditional_weight": float(conditional_weight),
        "jda_balance_source_classes": bool(balance_source_classes),
        "jda_change_history": "|".join(f"{value:.12g}" for value in solved.changes),
    }
    return JointDistributionResult(
        source_features=solved.source_latent.astype(np.float32),
        target_features=solved.target_latent.astype(np.float32),
        projection=solved.projection.astype(np.float32),
        feature_mean=prepared.mean.astype(np.float32),
        feature_scale=prepared.scale.astype(np.float32),
        classes=prepared.classes,
        target_assignments=labels,
        target_probabilities=solved.probabilities.astype(np.float32),
        eigenvalues=solved.eigenvalues,
        n_iterations=solved.iterations,
        converged=solved.converged,
        metadata=metadata,
    )
