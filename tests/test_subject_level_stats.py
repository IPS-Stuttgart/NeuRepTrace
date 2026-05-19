import pytest

from neureptrace.stats.subject_level import (
    exact_sign_test,
    one_sample_sign_flip_test,
    paired_sign_flip_test,
    permutation_p_value,
    summarize_against_chance,
)


def test_one_sample_sign_flip_test_uses_exact_enumeration_when_small():
    result = one_sample_sign_flip_test([1.0, 1.0, 1.0, 1.0], n_permutations=10_000)

    assert result.n_subjects == 4
    assert result.method == "exact_sign_flip"
    assert result.n_resamples == 16
    assert result.observed_mean == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.125)


def test_one_sample_sign_flip_test_supports_one_sided_greater():
    result = one_sample_sign_flip_test(
        [1.0, 1.0, 1.0, 1.0],
        alternative="greater",
        n_permutations=10_000,
    )

    assert result.p_value == pytest.approx(0.0625)


def test_paired_sign_flip_test_is_difference_level_alias():
    result = paired_sign_flip_test([0.2, 0.1, 0.3, 0.4], alternative="greater")

    assert result.null == 0.0
    assert result.observed_mean == pytest.approx(0.25)
    assert result.p_value == pytest.approx(0.0625)


def test_exact_sign_test_counts_nonzero_directions():
    result = exact_sign_test([1.0, 1.0, 1.0, -1.0])

    assert result.n_subjects == 4
    assert result.n_positive == 3
    assert result.n_negative == 1
    assert result.n_zero == 0
    assert result.p_value == pytest.approx(0.625)


def test_exact_sign_test_drops_zero_differences_by_default():
    result = exact_sign_test([0.7, 0.5, 0.4], null=0.5, alternative="greater")

    assert result.n_subjects == 2
    assert result.n_positive == 1
    assert result.n_negative == 1
    assert result.n_zero == 1
    assert result.p_value == pytest.approx(0.75)


def test_permutation_p_value_applies_plus_one_correction():
    p_value = permutation_p_value(2.0, [0.0, 1.0], alternative="greater")

    assert p_value == pytest.approx(1 / 3)


def test_summarize_against_chance_returns_reusable_fields():
    summary = summarize_against_chance(
        [0.6, 0.7, 0.8, 0.9],
        chance=0.5,
        metric="accuracy",
    )

    assert summary.metric == "accuracy"
    assert summary.n_subjects == 4
    assert summary.mean == pytest.approx(0.75)
    assert summary.mean_minus_chance == pytest.approx(0.25)
    assert summary.n_above_chance == 4
    assert summary.n_below_chance == 0
    assert summary.n_at_chance == 0
    assert summary.sign_flip_p == pytest.approx(0.0625)
    assert summary.sign_test_p == pytest.approx(0.0625)
    assert summary.to_dict()["metric"] == "accuracy"


def test_subject_level_helpers_reject_nan_by_default():
    with pytest.raises(ValueError, match="NaN"):
        summarize_against_chance([0.6, float("nan")], chance=0.5)


def test_subject_level_helpers_can_omit_nan_values():
    summary = summarize_against_chance([0.6, float("nan"), 0.8], chance=0.5, nan_policy="omit")

    assert summary.n_subjects == 2
    assert summary.mean == pytest.approx(0.7)
