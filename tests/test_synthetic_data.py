import numpy as np
import pytest

from neureptrace.synthetic_data import (
    SyntheticEpochConfig,
    make_synthetic_epochs,
    make_synthetic_participant_epochs,
    window_feature_matrix,
)


def test_make_synthetic_epochs_is_balanced_and_reproducible():
    config = SyntheticEpochConfig(
        n_classes=3,
        repeats_per_class=4,
        n_channels=5,
        n_times=41,
        random_seed=7,
        shuffle_trials=False,
    )

    first = make_synthetic_epochs(config)
    second = make_synthetic_epochs(config)

    assert first.data.shape == (12, 5, 41)
    assert first.labels.tolist() == [0, 1, 2] * 4
    assert first.groups.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert first.channel_names == ("MEG001", "MEG002", "MEG003", "MEG004", "MEG005")
    assert first.sensor_positions.shape == (5, 3)
    assert first.n_trials == 12
    assert first.n_channels == 5
    assert first.n_times == 41
    assert first.n_classes == 3
    assert np.allclose(first.data, second.data)
    assert np.array_equal(first.labels, second.labels)


def test_signal_window_features_have_stronger_class_structure_than_baseline():
    config = SyntheticEpochConfig(
        n_classes=4,
        repeats_per_class=6,
        n_channels=8,
        n_times=81,
        tmin=-0.2,
        tmax=0.5,
        signal_window=(0.1, 0.2),
        signal_scale=5.0,
        noise_scale=0.01,
        oscillation_scale=0.0,
        random_seed=11,
    )
    epochs = make_synthetic_epochs(config)

    signal_features = window_feature_matrix(epochs, config.signal_window)
    baseline_features = window_feature_matrix(epochs, (-0.18, -0.08))

    assert signal_features.shape == (24, 8)
    assert _mean_centroid_distance(signal_features, epochs.labels) > 20.0 * _mean_centroid_distance(
        baseline_features,
        epochs.labels,
    )


def test_make_synthetic_participant_epochs_share_labels_with_subject_specific_data():
    config = SyntheticEpochConfig(
        n_classes=3,
        repeats_per_class=3,
        n_channels=6,
        n_times=31,
        random_seed=13,
    )

    participants = make_synthetic_participant_epochs(
        2,
        config,
        transform_strength=0.5,
        participant_shift_scale=0.05,
    )

    assert [participant.participant_id for participant in participants] == ["sub-01", "sub-02"]
    assert np.array_equal(participants[0].labels, participants[1].labels)
    assert np.array_equal(participants[0].groups, participants[1].groups)
    assert participants[0].data.shape == participants[1].data.shape
    assert not np.allclose(participants[0].data, participants[1].data)


def test_window_feature_matrix_validates_window_and_reducer():
    epochs = make_synthetic_epochs(
        SyntheticEpochConfig(n_classes=2, repeats_per_class=2, n_channels=3, n_times=21)
    )

    flattened = window_feature_matrix(epochs, epochs.metadata["signal_window"], reducer="flatten")
    assert flattened.shape[0] == epochs.n_trials
    assert flattened.shape[1] > epochs.n_channels

    with pytest.raises(ValueError, match="window start"):
        window_feature_matrix(epochs, (0.2, 0.1))

    with pytest.raises(ValueError, match="overlap"):
        window_feature_matrix(epochs, (10.0, 11.0))

    with pytest.raises(ValueError, match="reducer"):
        window_feature_matrix(epochs, reducer="median")


def test_make_synthetic_participant_epochs_validates_controls():
    with pytest.raises(ValueError, match="n_participants"):
        make_synthetic_participant_epochs(0)

    with pytest.raises(ValueError, match="transform_strength"):
        make_synthetic_participant_epochs(1, transform_strength=1.5)

    with pytest.raises(ValueError, match="participant_shift_scale"):
        make_synthetic_participant_epochs(1, participant_shift_scale=-0.1)


def _mean_centroid_distance(features, labels):
    centroids = []
    for label in np.unique(labels):
        centroids.append(np.mean(features[labels == label], axis=0))
    distances = []
    for left_index, left in enumerate(centroids):
        for right in centroids[left_index + 1 :]:
            distances.append(float(np.linalg.norm(left - right)))
    return float(np.mean(distances))
