from __future__ import annotations

from neureptrace.mne_time_decode_ensemble import _parse_source_decoders
from neureptrace.observation_ensemble import ensemble_probability_observations


def _expect_unique_alias_error(call) -> None:
    try:
        call()
    except ValueError as exc:
        assert "unique after alias normalization" in str(exc)
    else:  # pragma: no cover - exercised only when the regression reappears
        raise AssertionError("Expected duplicate decoder aliases to raise ValueError.")


def test_time_decode_ensemble_rejects_duplicate_source_decoder_aliases() -> None:
    _expect_unique_alias_error(lambda: _parse_source_decoders(("linear_svm", "linear-svm")))


def test_observation_ensemble_rejects_duplicate_decoder_aliases() -> None:
    _expect_unique_alias_error(
        lambda: ensemble_probability_observations(
            object(),
            decoders=("linear_svm", "linear-svm"),
            baseline_window=None,
        )
    )
