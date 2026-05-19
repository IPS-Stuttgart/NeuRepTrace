"""Subject-level permutation and sign-test helpers.

These utilities are intentionally dataset-agnostic.  They operate on one
subject-level scalar per subject, for example an accuracy-minus-chance score or
paired decoder difference, and leave dataset-specific aggregation to callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike

Alternative = Literal["two-sided", "greater", "less"]
ExactMode = Literal["auto", "always", "never"]
NanPolicy = Literal["raise", "omit"]
ZeroMethod = Literal["drop", "positive", "negative"]

_VALID_ALTERNATIVES = {"two-sided", "greater", "less"}
_VALID_EXACT_MODES = {"auto", "always", "never"}
_VALID_NAN_POLICIES = {"raise", "omit"}
_VALID_ZERO_METHODS = {"drop", "positive", "negative"}


@dataclass(frozen=True)
class PermutationTestResult:
    """Result from a one-sample sign-flip permutation test."""

    n_subjects: int
    observed_mean: float
    null: float
    p_value: float
    alternative: Alternative
    method: str
    n_resamples: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a CSV/JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class SignTestResult:
    """Result from an exact binomial sign test."""

    n_subjects: int
    n_positive: int
    n_negative: int
    n_zero: int
    null: float
    p_value: float
    alternative: Alternative
    zero_method: ZeroMethod

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a CSV/JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class SubjectChanceSummary:
    """Compact summary of one subject-level metric against chance."""

    metric: str
    n_subjects: int
    chance: float
    mean: float
    median: float
    std: float
    mean_minus_chance: float
    median_minus_chance: float
    n_above_chance: int
    n_below_chance: int
    n_at_chance: int
    alternative: Alternative
    sign_flip_p: float
    sign_flip_method: str
    sign_test_p: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a CSV/JSON-friendly representation."""
        return asdict(self)


def _validate_choice(value: str, *, name: str, valid: set[str]) -> None:
    if value not in valid:
        expected = ", ".join(sorted(valid))
        raise ValueError(f"{name} must be one of {{{expected}}}; got {value!r}.")


def _as_1d_float_array(
    values: ArrayLike | Iterable[float],
    *,
    name: str = "values",
    min_values: int = 1,
    nan_policy: NanPolicy = "raise",
) -> np.ndarray:
    _validate_choice(nan_policy, name="nan_policy", valid=_VALID_NAN_POLICIES)
    if isinstance(values, np.ndarray):
        array = values.astype(float, copy=False)
    else:
        try:
            array = np.asarray(list(values), dtype=float)
        except TypeError as exc:
            raise ValueError(f"{name} must be one-dimensional.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values.")
    if np.isnan(array).any():
        if nan_policy == "raise":
            raise ValueError(f"{name} contains NaN values; pass nan_policy='omit' to drop them.")
        array = array[~np.isnan(array)]
    if len(array) < min_values:
        raise ValueError(f"Need at least {min_values} values; got {len(array)}.")
    return array


def permutation_p_value(
    observed: float,
    null_distribution: ArrayLike | Iterable[float],
    *,
    alternative: Alternative = "two-sided",
    correction: bool = True,
) -> float:
    """Return a permutation p-value from an observed statistic and null draws.

    Parameters
    ----------
    observed:
        Observed test statistic.
    null_distribution:
        One-dimensional array of null statistics.
    alternative:
        ``"greater"`` tests whether ``observed`` is unusually large,
        ``"less"`` tests whether it is unusually small, and ``"two-sided"``
        compares absolute deviations.
    correction:
        Apply the common ``+1`` finite Monte Carlo correction.  Disable this
        for exact, exhaustively enumerated null distributions.
    """
    _validate_choice(alternative, name="alternative", valid=_VALID_ALTERNATIVES)
    null = _as_1d_float_array(
        null_distribution,
        name="null_distribution",
        min_values=1,
        nan_policy="raise",
    )
    observed = float(observed)
    if not math.isfinite(observed):
        raise ValueError("observed must be finite.")

    if alternative == "greater":
        extreme = null >= observed
    elif alternative == "less":
        extreme = null <= observed
    else:
        extreme = np.abs(null) >= abs(observed)

    count = int(np.count_nonzero(extreme))
    if correction:
        return float((count + 1) / (len(null) + 1))
    return float(count / len(null))


def one_sample_sign_flip_test(
    values: ArrayLike | Iterable[float],
    *,
    null: float = 0.0,
    alternative: Alternative = "two-sided",
    n_permutations: int = 10_000,
    random_state: int | np.random.Generator | None = 13,
    exact: ExactMode = "auto",
    nan_policy: NanPolicy = "raise",
) -> PermutationTestResult:
    """Test whether subject-level values differ from ``null``.

    The test flips the sign of each subject's centered value.  For small sample
    sizes, all sign patterns are enumerated exactly by default; otherwise a
    seeded Monte Carlo approximation with the ``+1`` correction is used.
    """
    _validate_choice(alternative, name="alternative", valid=_VALID_ALTERNATIVES)
    _validate_choice(exact, name="exact", valid=_VALID_EXACT_MODES)
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    if not math.isfinite(null):
        raise ValueError("null must be finite.")

    sample = _as_1d_float_array(values, min_values=2, nan_policy=nan_policy)
    differences = sample - float(null)
    n_subjects = int(len(differences))
    observed_mean = float(differences.mean())
    exact_resamples = 2**n_subjects
    use_exact = exact == "always" or (exact == "auto" and exact_resamples <= n_permutations)

    if use_exact:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n_subjects)))
        null_means = signs @ differences / n_subjects
        p_value = permutation_p_value(
            observed_mean,
            null_means,
            alternative=alternative,
            correction=False,
        )
        return PermutationTestResult(
            n_subjects=n_subjects,
            observed_mean=observed_mean,
            null=float(null),
            p_value=p_value,
            alternative=alternative,
            method="exact_sign_flip",
            n_resamples=int(exact_resamples),
        )

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_permutations, n_subjects))
    null_means = signs @ differences / n_subjects
    p_value = permutation_p_value(
        observed_mean,
        null_means,
        alternative=alternative,
        correction=True,
    )
    return PermutationTestResult(
        n_subjects=n_subjects,
        observed_mean=observed_mean,
        null=float(null),
        p_value=p_value,
        alternative=alternative,
        method="monte_carlo_sign_flip",
        n_resamples=int(n_permutations),
    )


def paired_sign_flip_test(
    differences: ArrayLike | Iterable[float],
    *,
    alternative: Alternative = "two-sided",
    n_permutations: int = 10_000,
    random_state: int | np.random.Generator | None = 13,
    exact: ExactMode = "auto",
    nan_policy: NanPolicy = "raise",
) -> PermutationTestResult:
    """Run a sign-flip test on paired subject-level differences."""
    return one_sample_sign_flip_test(
        differences,
        null=0.0,
        alternative=alternative,
        n_permutations=n_permutations,
        random_state=random_state,
        exact=exact,
        nan_policy=nan_policy,
    )


def _binomial_pmf(k: int, n: int) -> float:
    return float(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)))


def _binomial_tail(k: int, n: int, *, alternative: Alternative) -> float:
    if n == 0:
        return 1.0
    if alternative == "greater":
        return float(math.fsum(_binomial_pmf(j, n) for j in range(k, n + 1)))
    if alternative == "less":
        return float(math.fsum(_binomial_pmf(j, n) for j in range(0, k + 1)))
    observed_probability = _binomial_pmf(k, n)
    return float(
        min(
            1.0,
            math.fsum(
                probability
                for probability in (_binomial_pmf(j, n) for j in range(n + 1))
                if probability <= observed_probability + 1e-15
            ),
        )
    )


def exact_sign_test(
    values: ArrayLike | Iterable[float],
    *,
    null: float = 0.0,
    alternative: Alternative = "two-sided",
    zero_method: ZeroMethod = "drop",
    nan_policy: NanPolicy = "raise",
) -> SignTestResult:
    """Run an exact binomial sign test on subject-level values."""
    _validate_choice(alternative, name="alternative", valid=_VALID_ALTERNATIVES)
    _validate_choice(zero_method, name="zero_method", valid=_VALID_ZERO_METHODS)
    if not math.isfinite(null):
        raise ValueError("null must be finite.")

    sample = _as_1d_float_array(values, min_values=1, nan_policy=nan_policy)
    differences = sample - float(null)
    n_zero = int(np.count_nonzero(differences == 0.0))
    if zero_method == "drop":
        tested = differences[differences != 0.0]
    elif zero_method == "positive":
        tested = differences.copy()
        tested[tested == 0.0] = 1.0
    else:
        tested = differences.copy()
        tested[tested == 0.0] = -1.0

    n_positive = int(np.count_nonzero(tested > 0.0))
    n_negative = int(np.count_nonzero(tested < 0.0))
    n_subjects = n_positive + n_negative
    p_value = _binomial_tail(n_positive, n_subjects, alternative=alternative)
    return SignTestResult(
        n_subjects=n_subjects,
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        null=float(null),
        p_value=p_value,
        alternative=alternative,
        zero_method=zero_method,
    )


def summarize_against_chance(
    values: ArrayLike | Iterable[float],
    *,
    chance: float,
    metric: str = "score",
    higher_is_better: bool = True,
    alternative: Alternative | None = None,
    n_permutations: int = 10_000,
    random_state: int | np.random.Generator | None = 13,
    exact: ExactMode = "auto",
    nan_policy: NanPolicy = "raise",
) -> SubjectChanceSummary:
    """Summarize one subject-level metric against a chance/null value."""
    if not math.isfinite(chance):
        raise ValueError("chance must be finite.")
    if alternative is None:
        alternative = "greater" if higher_is_better else "less"
    _validate_choice(alternative, name="alternative", valid=_VALID_ALTERNATIVES)

    sample = _as_1d_float_array(values, min_values=2, nan_policy=nan_policy)
    sign_flip = one_sample_sign_flip_test(
        sample,
        null=chance,
        alternative=alternative,
        n_permutations=n_permutations,
        random_state=random_state,
        exact=exact,
        nan_policy="raise",
    )
    sign_test = exact_sign_test(
        sample,
        null=chance,
        alternative=alternative,
        nan_policy="raise",
    )
    return SubjectChanceSummary(
        metric=str(metric),
        n_subjects=int(len(sample)),
        chance=float(chance),
        mean=float(sample.mean()),
        median=float(np.median(sample)),
        std=float(sample.std(ddof=1)) if len(sample) > 1 else 0.0,
        mean_minus_chance=float(sample.mean() - chance),
        median_minus_chance=float(np.median(sample) - chance),
        n_above_chance=int(np.count_nonzero(sample > chance)),
        n_below_chance=int(np.count_nonzero(sample < chance)),
        n_at_chance=int(np.count_nonzero(sample == chance)),
        alternative=alternative,
        sign_flip_p=sign_flip.p_value,
        sign_flip_method=sign_flip.method,
        sign_test_p=sign_test.p_value,
    )


__all__ = [
    "PermutationTestResult",
    "SignTestResult",
    "SubjectChanceSummary",
    "exact_sign_test",
    "one_sample_sign_flip_test",
    "paired_sign_flip_test",
    "permutation_p_value",
    "summarize_against_chance",
]
