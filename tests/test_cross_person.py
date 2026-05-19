import numpy as np

from neureptrace.cross_person import (
    CrossPersonCandidate,
    SubjectFeatureMatrix,
    run_nested_cross_person_from_loader,
)


def _synthetic_subject(subject, *, rotation=0.0):
    rng = np.random.default_rng(abs(hash(subject)) % (2**32))
    labels = np.repeat(np.array([0, 1]), 12)
    base = np.column_stack([labels * 2.0 - 1.0, 1.0 - labels * 2.0])
    noise = rng.normal(scale=0.15, size=base.shape)
    features = base + noise
    if rotation:
        c = np.cos(rotation)
        s = np.sin(rotation)
        features = features @ np.array([[c, -s], [s, c]])
    return SubjectFeatureMatrix(subject=subject, features=features, labels=labels)


def test_nested_cross_person_decoding_beats_chance_on_shared_signal():
    data = {
        "s1": _synthetic_subject("s1"),
        "s2": _synthetic_subject("s2"),
        "s3": _synthetic_subject("s3"),
        "s4": _synthetic_subject("s4"),
    }

    def loader(subject, candidate):
        del candidate
        return data[subject]

    candidate = CrossPersonCandidate(
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        normalization="none",
        alignment="none",
    )
    artifacts = run_nested_cross_person_from_loader(
        tuple(data),
        candidate_configs=(candidate,),
        feature_loader=loader,
        selection_ensemble_size=1,
    )
    assert len(artifacts.outer) == 4
    assert np.mean([row["balanced_accuracy"] for row in artifacts.outer]) > 0.90
    assert len(artifacts.predictions) == 4 * 24


def test_topk_window_diversity_selects_multiple_candidates():
    data = {
        "s1": _synthetic_subject("s1"),
        "s2": _synthetic_subject("s2"),
        "s3": _synthetic_subject("s3"),
        "s4": _synthetic_subject("s4"),
    }

    def loader(subject, candidate):
        del candidate
        return data[subject]

    candidates = (
        CrossPersonCandidate(window_center=0.150, decoder="logistic", emission_mode="uncalibrated", feature_preprocessor="none", pca_components=None, normalization="none"),
        CrossPersonCandidate(window_center=0.200, decoder="logistic", emission_mode="uncalibrated", feature_preprocessor="none", pca_components=None, normalization="none"),
    )
    artifacts = run_nested_cross_person_from_loader(
        tuple(data),
        candidate_configs=candidates,
        feature_loader=loader,
        selection_ensemble_size=2,
        selection_ensemble_diversity="window",
    )
    assert all(row["selection_ensemble_size"] == 2 for row in artifacts.outer)
